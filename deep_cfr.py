import torch
import torch.nn.functional as F
import numpy as np
import random
import os
import copy
import logging
import argparse
from collections import defaultdict, deque
from torch.utils.tensorboard import SummaryWriter
from game_env import PokerEnv, HandProcessor
from models import PokerNet, CFRNetwork

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deep_cfr.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DeepCFR:
    def __init__(self):
        self.env = PokerEnv()
        self.hand_processor = HandProcessor()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Initialize networks
        self.advantage_net = CFRNetwork(input_dim=2, device=self.device)
        self.strategy_net = CFRNetwork(input_dim=2, device=self.device)
        self.opponent_net = CFRNetwork(input_dim=2, device=self.device)

        # Training parameters
        self.epsilon = 0.15
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        self.iterations = 0

        # Memory buffers
        self.advantage_memory = deque(maxlen=1000000)
        self.cumulative_regrets = defaultdict(lambda: np.zeros(5))
        self.cumulative_strategy = defaultdict(lambda: {'counts': np.zeros(5), 'visits': 0})

        self.training_agent = 'player_0'
        self.writer = SummaryWriter()
        self.save_dir = "checkpoints"
        os.makedirs(self.save_dir, exist_ok=True)

    def encode_state(self, observation):
        """Convert observation to (hand_strength, pot_size) tuple"""
        try:
            # Extract card information
            hole_mask = observation[:52]
            hole_indices = np.where(hole_mask == 1)[0] + 1
            
            # Community cards (flop + turn + river)
            board_mask = observation[52:52+52*5]
            board_indices = np.where(board_mask == 1)[0] + 1
            
            hand_strength = self.hand_processor.get_strength(hole_indices, board_indices)
            pot_size = observation[52] + observation[53]  # Total chips in pot
            return (round(hand_strength, 2), round(pot_size, 1))
        except Exception as e:
            logger.error(f"Error encoding state: {str(e)}")
            raise

    def encode_state_raw(self, observation):
        try:
            hand_strength, pot_size = self.encode_state(observation)
            return torch.FloatTensor([hand_strength, pot_size]).to(self.device)
        except Exception as e:
            logger.error(f"Error in encode_state_raw: {str(e)}")
            raise

    def get_action_probs(self, net, state_tensor, legal_mask):
        try:
            with torch.no_grad():
                logits = net.net(state_tensor.unsqueeze(0))
                probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
                
                if (legal_mask == 0).all():
                    logger.warning("No legal actions available!")
                    return legal_mask
                
                legal_probs = probs * legal_mask
                sum_probs = legal_probs.sum()
                if sum_probs == 0:
                    legal_probs = legal_mask / legal_mask.sum()
                else:
                    legal_probs /= sum_probs
                return legal_probs
        except Exception as e:
            logger.error(f"Error getting action probs: {str(e)}")
            raise

    def _cfr_iteration(self):
        try:
            self.env.reset()
            history = []
            rp_self, rp_opp = 1.0, 1.0
            logger.debug(f"\nStarting CFR iteration {self.iterations}")

            for agent in self.env.env.agent_iter():
                last_output = self.env.env.last()
                obs_dict, reward, termination, truncation, _ = last_output
                done = termination or truncation
                
                if done:
                    action = None
                else:
                    obs = obs_dict['observation']
                    legal_mask = obs_dict['action_mask']
                    state_tensor = self.encode_state_raw(obs)
                    
                    logger.debug(f"Agent: {agent} | Legal mask: {legal_mask}")
                    logger.debug(f"State tensor: {state_tensor.cpu().numpy()}")

                    if agent == self.training_agent:
                        if random.random() < self.epsilon:
                            strategy = legal_mask / legal_mask.sum()
                        else:
                            strategy = self.get_action_probs(self.strategy_net, state_tensor, legal_mask)
                        
                        action = np.random.choice(len(strategy), p=strategy)
                        history.append((state_tensor, strategy, action, rp_self, rp_opp))
                        rp_self *= strategy[action]
                        logger.debug(f"Training agent action: {action} | Strategy: {strategy}")
                    else:
                        strategy = self.get_action_probs(self.opponent_net, state_tensor, legal_mask)
                        action = np.random.choice(len(strategy), p=strategy)
                        rp_opp *= strategy[action]
                        logger.debug(f"Opponent action: {action} | Strategy: {strategy}")

                self.env.env.step(action if not done else None)

            final_rewards = {agent: self.env.env.rewards.get(agent, 0) for agent in self.env.agents}
            logger.debug(f"Final rewards: {final_rewards}")
            self._update_regrets(history, final_rewards)

        except Exception as e:
            logger.error(f"Error in CFR iteration: {str(e)}", exc_info=True)
            raise

    def _update_regrets(self, history, final_rewards):
        try:
            player_reward = final_rewards.get(self.training_agent, 0)
            logger.info(f"Updating regrets with {len(history)} history entries")

            for state_tensor, strategy, action, rp_self, rp_opp in history:
                state_key = tuple(state_tensor.cpu().numpy().round(2))
                cf_value = player_reward * (rp_opp / (rp_self + 1e-8))
                
                immediate_regret = np.zeros(5)
                immediate_regret[action] = cf_value * (1 - strategy[action])
                for a in range(5):
                    if a != action:
                        immediate_regret[a] = -cf_value * strategy[a]
                
                self.cumulative_regrets[state_key] += immediate_regret
                self.cumulative_regrets[state_key] = np.maximum(self.cumulative_regrets[state_key], 0)
                
                self.advantage_memory.append((state_tensor.cpu().numpy(), self.cumulative_regrets[state_key]))
                
                self.cumulative_strategy[state_key]['counts'] += strategy * rp_self
                self.cumulative_strategy[state_key]['visits'] += 1

        except Exception as e:
            logger.error(f"Error updating regrets: {str(e)}", exc_info=True)
            raise

    def _update_networks(self, batch_size):
        try:
            if len(self.advantage_memory) < batch_size:
                return 0.0
            
            batch = random.sample(self.advantage_memory, batch_size)
            states, regrets = zip(*batch)
            
            states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
            regrets_tensor = torch.FloatTensor(np.array(regrets)).to(self.device)
            
            self.advantage_net.optimizer.zero_grad()
            pred_regrets = self.advantage_net.net(states_tensor)
            loss = F.mse_loss(pred_regrets, regrets_tensor)
            loss.backward()
            self.advantage_net.optimizer.step()
            
            logger.debug(f"Advantage network loss: {loss.item():.4f}")
            return loss.item()

        except Exception as e:
            logger.error(f"Error updating networks: {str(e)}", exc_info=True)
            raise

    def _update_strategy_net(self, batch_size):
        try:
            if not self.cumulative_strategy:
                return 0.0
            
            keys = list(self.cumulative_strategy.keys())
            if not keys:
                return 0.0
                
            sample_size = min(batch_size, len(keys))
            indices = np.random.choice(len(keys), size=sample_size, replace=False)
            sampled_keys = [keys[i] for i in indices]
            
            states = []
            targets = []
            for key in sampled_keys:
                strategy = self.cumulative_strategy[key]['counts'] / self.cumulative_strategy[key]['visits']
                states.append(torch.FloatTensor([key[0], key[1]]))
                targets.append(strategy)
            
            states_tensor = torch.stack(states).to(self.device)
            targets_tensor = torch.FloatTensor(np.array(targets)).to(self.device)
            
            self.strategy_net.optimizer.zero_grad()
            logits = self.strategy_net.net(states_tensor)
            loss = F.kl_div(F.log_softmax(logits, dim=-1), F.softmax(targets_tensor, dim=-1))
            loss.backward()
            self.strategy_net.optimizer.step()
            
            logger.debug(f"Strategy network loss: {loss.item():.4f}")
            return loss.item()

        except Exception as e:
            logger.error(f"Error updating strategy network: {str(e)}", exc_info=True)
            raise

    def train(self, iterations=10000, batch_size=512, save_interval=1000):
        try:
            logger.info(f"Starting training for {iterations} iterations")
            for _ in range(iterations):
                self.iterations += 1
                self._cfr_iteration()
                
                # Update networks
                loss_adv = self._update_networks(batch_size)
                loss_strat = self._update_strategy_net(batch_size)
                
                # Update opponent network
                if self.iterations % 100 == 0:
                    self.opponent_net.update_target(tau=0.01)
                    logger.debug("Updated opponent network")
                
                # Decay exploration
                self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
                
                # Logging
                self.writer.add_scalar('Loss/Advantage', loss_adv, self.iterations)
                self.writer.add_scalar('Loss/Strategy', loss_strat, self.iterations)
                self.writer.add_scalar('Params/Epsilon', self.epsilon, self.iterations)
                
                if self.iterations % save_interval == 0:
                    self._save_checkpoint()
                    logger.info(f"Saved checkpoint at iteration {self.iterations}")
            
            self.writer.close()
            logger.info("Training completed successfully")

        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            raise

    def _save_checkpoint(self):
        checkpoint = {
            'iteration': self.iterations,
            'advantage_net': self.advantage_net.net.state_dict(),
            'strategy_net': self.strategy_net.net.state_dict(),
            'optimizer_adv': self.advantage_net.optimizer.state_dict(),
            'optimizer_strat': self.strategy_net.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }
        torch.save(checkpoint, os.path.join(self.save_dir, f'checkpoint_{self.iterations}.pth'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    args = parser.parse_args()
    logger.setLevel(args.log)
    
    try:
        trainer = DeepCFR()
        trainer.train(iterations=10000, batch_size=512, save_interval=1000)
    except Exception as e:
        logger.critical(f"Critical error in main execution: {str(e)}", exc_info=True)