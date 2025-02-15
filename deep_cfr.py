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
        
        # Initialize networks
        self.advantage_net = CFRNetwork(self.device)
        self.strategy_net = CFRNetwork(self.device)
        
        # Opponent network uses a rolling average (via target_net)
        self.opponent_net = CFRNetwork(self.device)
        
        # Training parameters
        self.epsilon = 0.15
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        self.iterations = 0
        
        # Memory buffers for regret training
        self.advantage_memory = deque(maxlen=1000000)
        self.cumulative_regrets = defaultdict(lambda: np.zeros(5))
        self.cumulative_strategy = defaultdict(lambda: np.zeros(5))
        
        # Define the training agent (assumed to be the first agent)
        self.training_agent = self.env.agents[0]
        
        # Logging
        self.writer = SummaryWriter()  # Logs saved in "runs"
        self.save_dir = "checkpoints"
        os.makedirs(self.save_dir, exist_ok=True)

    def encode_state(self, observation):
        # State abstraction: use hand strength and pot size to generate a key.
        # Adapt indices as needed based on your observation structure.
        hand_strength = self.hand_processor.get_strength(observation[:2], observation[2:7])
        pot_size = observation[52] if len(observation) > 52 else 0.0
        key_str = f"{hand_strength:.2f}_{pot_size:.1f}"
        return hash(key_str)

    def encode_state_raw(self, observation):
        # Convert the raw observation into a tensor for network input.
        return torch.FloatTensor(observation).to(self.device)

    def get_action_probs(self, net, state_tensor, legal_mask):
        logits, _ = net.net(state_tensor.unsqueeze(0))
        probs = F.softmax(logits, dim=-1).detach().squeeze().cpu().numpy()
        probs = probs * legal_mask + 1e-8
        return probs / probs.sum()

    def train(self, iterations=100000, batch_size=512, save_interval=20000):
        for _ in range(iterations):
            self.iterations += 1
            self._cfr_iteration()
            
            if len(self.advantage_memory) >= batch_size:
                loss_adv = self._update_networks(batch_size)
                self.writer.add_scalar('Loss/Advantage', loss_adv, self.iterations)
            
            if self.iterations % 500 == 0 and self.cumulative_strategy:
                loss_strat = self._update_strategy_net(batch_size)
                self.writer.add_scalar('Loss/Strategy', loss_strat, self.iterations)
            
            self.writer.add_scalar('Metrics/Epsilon', self.epsilon, self.iterations)
            self.writer.add_scalar('Metrics/MemorySize', len(self.advantage_memory), self.iterations)
            for i, param_group in enumerate(self.advantage_net.optimizer.param_groups):
                self.writer.add_scalar(f'LearningRate/Advantage/group_{i}', param_group['lr'], self.iterations)
            
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
            if self.iterations % 100 == 0:
                win_rate = self.validate()
                self.writer.add_scalar('Metrics/WinRate', win_rate, self.iterations)
                print(f"Iter {self.iterations}: WR={win_rate:.2f} | ε={self.epsilon:.3f}")
            
            if self.iterations % save_interval == 0:
                self._save_checkpoint()
                # Log network histograms
                for net_name, net in zip(['AdvantageNet', 'StrategyNet'], [self.advantage_net.net, self.strategy_net.net]):
                    for name, param in net.named_parameters():
                        self.writer.add_histogram(f'{net_name}/{name}', param, self.iterations)
                        if param.grad is not None:
                            self.writer.add_histogram(f'{net_name}/{name}_grad', param.grad, self.iterations)
            
            # Rolling update of opponent using soft target update
            if self.iterations % 100 == 0:
                self.opponent_net.update_target(tau=0.001)
    
    def _cfr_iteration(self):
        self.env.reset()
        history = []  # Will store tuples: (state, strategy, action, reach_prob)
        reach_prob = 1.0  # Initialize reach probability
        for agent in self.env.env.agent_iter():
            obs_dict, rewards = self.env._get_observations()
            obs = obs_dict[agent]
            if obs['done']:
                action = None
            else:
                state = self.encode_state_raw(obs['observation'])
                legal_mask = obs['action_mask']
                if agent == self.training_agent:
                    if random.random() < self.epsilon:
                        strategy = legal_mask / legal_mask.sum()
                    else:
                        strategy = self.get_action_probs(self.strategy_net, state, legal_mask)
                    action = np.random.choice(5, p=strategy)
                    history.append((state, strategy, action, reach_prob))
                else:
                    if self.opponent_net is not None:
                        strategy = self.get_action_probs(self.opponent_net, state, legal_mask)
                    else:
                        strategy = legal_mask / legal_mask.sum()
                    action = np.random.choice(5, p=strategy)
                    # Update reach probability using opponent's action probability
                    reach_prob *= strategy[action]
            self.env.env.step(action if not obs['done'] else None)
        final_rewards = {agent: self.env.env.rewards.get(agent, 0) for agent in self.env.agents}
        self._update_regrets(history, final_rewards)
    
    def _update_regrets(self, history, final_rewards):
        for state, strategy, action, reach_prob in history:
            state_key = self.encode_state(state.cpu().numpy())  # use abstracted key
            # Approximate CFV using reach probability
            cf_value = final_rewards[self.training_agent] / (1 - reach_prob + 1e-8)
            immediate_regret = cf_value * (np.eye(5)[action] - strategy)
            self.advantage_memory.append((state.cpu().numpy(), immediate_regret))
            self.cumulative_regrets[state_key] += immediate_regret
            self.cumulative_regrets[state_key] = np.maximum(self.cumulative_regrets[state_key], 0)
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
    
    def _update_strategy_net(self, batch_size):
        if not self.cumulative_strategy:
            return 0.0
        keys = list(self.cumulative_strategy.keys())
        sampled_keys = random.sample(keys, min(batch_size, len(keys)))
        state_tensors = []
        targets = []
        for key in sampled_keys:
            # In practice, you’d want a more sophisticated state representation.
            # Here we use the key (already a hash) as a proxy. This is a placeholder.
            state_array = np.array([float(key % 1000)])  # crude placeholder conversion
            state_tensor = torch.FloatTensor(state_array).to(self.device)
            cumulative = self.cumulative_strategy[key]
            total = cumulative.sum()
            target = cumulative / total if total > 0 else np.ones_like(cumulative) / len(cumulative)
            state_tensors.append(state_tensor)
            targets.append(target)
        states_batch = torch.stack(state_tensors)
        targets_batch = torch.FloatTensor(np.array(targets)).to(self.device)
        self.strategy_net.optimizer.zero_grad()
        logits, _ = self.strategy_net.net(states_batch)
        probs = F.softmax(logits, dim=-1)
        loss = F.kl_div(torch.log(probs + 1e-8), targets_batch, reduction='batchmean')
        loss.backward()
        self.strategy_net.optimizer.step()
        return loss.item()
    
    def validate(self, num_games=100):
        wins = 0
        self.strategy_net.net.eval()
        self.opponent_net.net.eval()
        for _ in range(num_games):
            self.env.reset()
            for agent in self.env.env.agent_iter():
                obs_dict, _ = self.env._get_observations()
                obs = obs_dict[agent]
                if obs['done']:
                    action = None
                else:
                    with torch.no_grad():
                        state = self.encode_state_raw(obs['observation'])
                        legal_mask = obs['action_mask']
                        if agent == self.training_agent:
                            probs = self.get_action_probs(self.strategy_net, state, legal_mask)
                        else:
                            probs = self.get_action_probs(self.opponent_net, state, legal_mask)
                        action = np.random.choice(5, p=probs)
                self.env.env.step(action)
            wins += int(self.env.env.rewards.get(self.training_agent, 0) > 0)
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