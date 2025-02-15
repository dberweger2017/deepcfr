import torch
import torch.nn.functional as F
import numpy as np
import random
import os
import copy
from collections import defaultdict, deque
from torch.utils.tensorboard import SummaryWriter
from game_env import PokerEnv, HandProcessor
from models import PokerNet, CFRNetwork

class DeepCFR:
    def __init__(self):
        self.env = PokerEnv()
        self.hand_processor = HandProcessor()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks with correct input dimensions
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
        
        self.training_agent = 'player_0'  # PettingZoo agent name
        self.writer = SummaryWriter()
        self.save_dir = "checkpoints"
        os.makedirs(self.save_dir, exist_ok=True)

    def encode_state(self, observation):
        """Convert observation to (hand_strength, pot_size) tuple"""
        # Extract card indices from one-hot encoding
        hole_mask = observation[:52]
        hole_indices = np.where(hole_mask == 1)[0] + 1  # Convert to 1-based indices
        
        # Community cards (flop + turn + river)
        board_mask = observation[52:52+52*5]
        board_indices = np.where(board_mask == 1)[0] + 1
        
        hand_strength = self.hand_processor.get_strength(hole_indices, board_indices)
        pot_size = observation[52] + observation[53]  # Total chips in pot
        return (round(hand_strength, 2), round(pot_size, 1))

    def encode_state_raw(self, observation):
        hand_strength, pot_size = self.encode_state(observation)
        return torch.FloatTensor([hand_strength, pot_size]).to(self.device)

    def get_action_probs(self, net, state_tensor, legal_mask):
        with torch.no_grad():
            logits = net.net(state_tensor.unsqueeze(0))
            probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
            legal_probs = probs * legal_mask
            if legal_probs.sum() == 0:
                legal_probs = legal_mask / legal_mask.sum()
            else:
                legal_probs /= legal_probs.sum()
            return legal_probs

    def _cfr_iteration(self):
        self.env.reset()
        history = []
        rp_self, rp_opp = 1.0, 1.0
        
        for agent in self.env.env.agent_iter():
            obs_dict, _, _, _ = self.env.env.last()
            obs = obs_dict['observation'] if agent == self.training_agent else obs_dict['observation']
            legal_mask = obs_dict['action_mask']
            
            if obs_dict['done']:
                action = None
            else:
                state_tensor = self.encode_state_raw(obs)
                
                if agent == self.training_agent:
                    # Epsilon-greedy exploration
                    if random.random() < self.epsilon:
                        strategy = legal_mask / legal_mask.sum()
                    else:
                        strategy = self.get_action_probs(self.strategy_net, state_tensor, legal_mask)
                    
                    action = np.random.choice(len(strategy), p=strategy)
                    history.append((state_tensor, strategy, action, rp_self, rp_opp))
                    rp_self *= strategy[action]
                else:
                    # Use opponent net for other players
                    strategy = self.get_action_probs(self.opponent_net, state_tensor, legal_mask)
                    action = np.random.choice(len(strategy), p=strategy)
                    rp_opp *= strategy[action]

            self.env.env.step(action if not obs_dict['done'] else None)
        
        # Update with final payoffs
        final_rewards = {agent: self.env.env.rewards[agent] for agent in self.env.agents}
        self._update_regrets(history, final_rewards)

    def _update_regrets(self, history, final_rewards):
        player_reward = final_rewards[self.training_agent]
        
        for state_tensor, strategy, action, rp_self, rp_opp in history:
            state_key = self.encode_state(state_tensor.cpu().numpy())
            cf_value = player_reward * (rp_opp / (rp_self + 1e-8))
            
            # Calculate immediate regret
            immediate_regret = np.zeros(5)
            immediate_regret[action] = cf_value * (1 - strategy[action])
            for a in range(5):
                if a != action:
                    immediate_regret[a] = -cf_value * strategy[a]
            
            # Update cumulative regrets
            self.cumulative_regrets[state_key] += immediate_regret
            self.cumulative_regrets[state_key] = np.maximum(self.cumulative_regrets[state_key], 0)
            
            # Store in advantage memory
            self.advantage_memory.append((state_tensor.cpu().numpy(), self.cumulative_regrets[state_key]))
            
            # Update cumulative strategy
            self.cumulative_strategy[state_key]['counts'] += strategy * rp_self
            self.cumulative_strategy[state_key]['visits'] += 1

    def _update_networks(self, batch_size):
        if len(self.advantage_memory) < batch_size:
            return 0.0
        
        batch = random.sample(self.advantage_memory, batch_size)
        states, regrets = zip(*batch)
        
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        regrets_tensor = torch.FloatTensor(np.array(regrets)).to(self.device)
        
        # Update advantage network
        self.advantage_net.optimizer.zero_grad()
        pred_regrets = self.advantage_net.net(states_tensor)
        loss = F.mse_loss(pred_regrets, regrets_tensor)
        loss.backward()
        self.advantage_net.optimizer.step()
        
        return loss.item()

    def _update_strategy_net(self, batch_size):
        if not self.cumulative_strategy:
            return 0.0
        
        # Sample from cumulative strategy
        keys = list(self.cumulative_strategy.keys())
        sampled_keys = np.random.choice(keys, size=min(batch_size, len(keys)), replace=False)
        
        states = []
        targets = []
        for key in sampled_keys:
            strategy = self.cumulative_strategy[key]['counts'] / self.cumulative_strategy[key]['visits']
            states.append(torch.FloatTensor([key[0], key[1]]))
            targets.append(strategy)
        
        states_tensor = torch.stack(states).to(self.device)
        targets_tensor = torch.FloatTensor(np.array(targets)).to(self.device)
        
        # Update strategy network
        self.strategy_net.optimizer.zero_grad()
        logits = self.strategy_net.net(states_tensor)
        loss = F.kl_div(F.log_softmax(logits, dim=-1), F.softmax(targets_tensor, dim=-1))
        loss.backward()
        self.strategy_net.optimizer.step()
        
        return loss.item()

    def train(self, iterations=10000, batch_size=512, save_interval=1000):
        for _ in range(iterations):
            self.iterations += 1
            self._cfr_iteration()
            
            # Update networks
            loss_adv = self._update_networks(batch_size)
            loss_strat = self._update_strategy_net(batch_size)
            
            # Update opponent network
            if self.iterations % 100 == 0:
                self.opponent_net.update_target(self.strategy_net.net, tau=0.01)
            
            # Decay exploration
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
            # Logging
            self.writer.add_scalar('Loss/Advantage', loss_adv, self.iterations)
            self.writer.add_scalar('Loss/Strategy', loss_strat, self.iterations)
            self.writer.add_scalar('Params/Epsilon', self.epsilon, self.iterations)
            
            if self.iterations % save_interval == 0:
                self._save_checkpoint()
        
        self.writer.close()

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
    trainer = DeepCFR()
    trainer.train(iterations=10000, batch_size=512, save_interval=1000)