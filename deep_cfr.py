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
        
        # Initialize networks for training.
        # The advantage network is updated via regret matching,
        # while the strategy network is used for selecting actions.
        self.advantage_net = CFRNetwork(self.device)
        self.strategy_net = CFRNetwork(self.device)
        
        # Opponent network (frozen opponent), initially not set (random opponent)
        self.opponent_net = None
        
        # Training parameters
        self.epsilon = 0.15
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        self.iterations = 0
        
        # Memory buffers for regret training
        self.advantage_memory = deque(maxlen=1000000)
        self.cumulative_regrets = defaultdict(lambda: np.zeros(5))
        self.cumulative_strategy = defaultdict(lambda: np.zeros(5))
        
        # Define which agent is training (assume first agent in the environment)
        self.training_agent = self.env.agents[0]
        
        # Logging
        self.writer = SummaryWriter()
        self.save_dir = "checkpoints"
        os.makedirs(self.save_dir, exist_ok=True)

    def encode_state(self, observation):
        return torch.FloatTensor(observation).to(self.device)

    def get_action_probs(self, net, state, legal_mask):
        logits, _ = net.net(state.unsqueeze(0))
        probs = F.softmax(logits, dim=-1).detach().squeeze().cpu().numpy()
        probs = probs * legal_mask + 1e-8
        return probs / probs.sum()

    def train(self, iterations=10000, batch_size=512, save_interval=500):
        for _ in range(iterations):
            self.iterations += 1
            self._cfr_iteration()
            
            if len(self.advantage_memory) >= batch_size:
                loss = self._update_networks(batch_size)
                self.writer.add_scalar('Loss/Advantage', loss, self.iterations)
            
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
            if self.iterations % 100 == 0:
                win_rate = self.validate()
                self.writer.add_scalar('Metrics/WinRate', win_rate, self.iterations)
                print(f"Iter {self.iterations}: WR={win_rate:.2f} | ε={self.epsilon:.3f}")
            
            if self.iterations % save_interval == 0:
                self._save_checkpoint()

            # Update opponent model every 10,000 iterations.
            # This freezes the current strategy_net as the opponent.
            if self.iterations % 10000 == 0:
                self.opponent_net = copy.deepcopy(self.strategy_net)
                print(f"Opponent model updated at iteration {self.iterations}")

    def _cfr_iteration(self):
        self.env.reset()
        # We'll record history only for the training agent.
        history = []  # list of tuples: (state, probs, action)
        
        # Loop through agents in the environment.
        for agent in self.env.env.agent_iter():
            obs_dict, rewards = self.env._get_observations()
            obs = obs_dict[agent]
            
            if obs['done']:
                action = None
            else:
                state = self.encode_state(obs['observation'])
                legal_mask = obs['action_mask']
                
                if agent == self.training_agent:
                    # Training agent: use epsilon-greedy with current strategy_net.
                    if random.random() < self.epsilon:
                        probs = legal_mask / legal_mask.sum()
                    else:
                        probs = self.get_action_probs(self.strategy_net, state, legal_mask)
                    action = np.random.choice(5, p=probs)
                    history.append((state, probs, action))
                else:
                    # Opponent agent: use the frozen opponent_net if available,
                    # otherwise act randomly.
                    if self.opponent_net is not None:
                        probs = self.get_action_probs(self.opponent_net, state, legal_mask)
                    else:
                        probs = legal_mask / legal_mask.sum()
                    action = np.random.choice(5, p=probs)
            
            self.env.env.step(action if not obs['done'] else None)
        
        # Process final rewards and update regrets only for the training agent.
        final_rewards = {agent: self.env.env.rewards.get(agent, 0) for agent in self.env.agents}
        self._update_regrets(history, final_rewards)

    def _update_regrets(self, history, final_rewards):
        # Update regrets using experiences from the training agent.
        for state, strategy, action in history:
            state_key = tuple(state.cpu().numpy().tolist())
            cf_value = final_rewards[self.training_agent] - np.mean(list(final_rewards.values()))
            
            # Compute regret: positive regret for not having taken each action.
            regret = cf_value * (np.eye(5)[action] - strategy)
            self.cumulative_regrets[state_key] += regret
            self.cumulative_regrets[state_key] = np.maximum(self.cumulative_regrets[state_key], 0)
            
            # Save the experience for network training.
            self.advantage_memory.append((state.cpu().numpy(), self.cumulative_regrets[state_key].copy()))
            self.cumulative_strategy[state_key] += strategy

    def _update_networks(self, batch_size):
        batch = random.sample(self.advantage_memory, batch_size)
        states, regrets = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        regrets = torch.FloatTensor(np.array(regrets)).to(self.device)
        
        self.advantage_net.optimizer.zero_grad()
        policy_out, _ = self.advantage_net.net(states)
        loss = F.mse_loss(policy_out, regrets)
        loss.backward()
        self.advantage_net.optimizer.step()
        
        return loss.item()

    def validate(self, num_games=100):
        wins = 0
        self.strategy_net.net.eval()
        if self.opponent_net is not None:
            self.opponent_net.net.eval()
        
        for _ in range(num_games):
            self.env.reset()
            for agent in self.env.env.agent_iter():
                obs_dict, _ = self.env._get_observations()
                obs = obs_dict[agent]
                
                if obs['done']:
                    continue
                
                with torch.no_grad():
                    state = self.encode_state(obs['observation'])
                    legal_mask = obs['action_mask']
                    if agent == self.training_agent:
                        probs = self.get_action_probs(self.strategy_net, state, legal_mask)
                    else:
                        if self.opponent_net is not None:
                            probs = self.get_action_probs(self.opponent_net, state, legal_mask)
                        else:
                            probs = legal_mask / legal_mask.sum()
                    action = np.random.choice(5, p=probs)
                
                self.env.env.step(action)
            
            wins += int(self.env.env.rewards[self.training_agent] > 0)
        
        self.strategy_net.net.train()
        return wins / num_games

    def _save_checkpoint(self):
        path = os.path.join(self.save_dir, f"checkpoint_{self.iterations}.pth")
        torch.save({
            'iteration': self.iterations,
            'advantage_net': self.advantage_net.net.state_dict(),
            'strategy_net': self.strategy_net.net.state_dict(),
            'optimizer_adv': self.advantage_net.optimizer.state_dict(),
            'optimizer_strat': self.strategy_net.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.advantage_net.net.load_state_dict(checkpoint['advantage_net'])
        self.strategy_net.net.load_state_dict(checkpoint['strategy_net'])
        self.advantage_net.optimizer.load_state_dict(checkpoint['optimizer_adv'])
        self.strategy_net.optimizer.load_state_dict(checkpoint['optimizer_strat'])
        self.epsilon = checkpoint['epsilon']
        self.iterations = checkpoint['iteration']

if __name__ == "__main__":
    trainer = DeepCFR()
    trainer.train(iterations=10000, batch_size=512, save_interval=500)