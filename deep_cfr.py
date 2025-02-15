# deep_cfr.py
import torch
import torch.nn.functional as F
import numpy as np
import random
import os
from collections import defaultdict, deque
from torch.utils.tensorboard import SummaryWriter
from models import NLHEPolicyNetwork
from game_env import NLHEEnvironment, HandProcessor

class DeepCFR:
    def __init__(self):
        self.env = NLHEEnvironment()
        self.hand_processor = HandProcessor()
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.advantage_net = NLHEPolicyNetwork().to(self.device)
        self.strategy_net = NLHEPolicyNetwork().to(self.device)
        
        # Optimization
        self.optimizer = torch.optim.Adam([
            {'params': self.advantage_net.parameters(), 'lr': 1e-4},
            {'params': self.strategy_net.parameters(), 'lr': 5e-5}
        ])
        
        # Training state
        self.epsilon = 0.1
        self.epsilon_decay = 0.999
        self.iteration = 0
        self.advantage_memory = deque(maxlen=100000)
        self.strategy_memory = deque(maxlen=100000)
        self.cumulative_regrets = defaultdict(lambda: np.zeros(3))
        self.cumulative_strategy = defaultdict(lambda: np.zeros(3))

        # Logging
        self.writer = SummaryWriter()
        self.save_dir = "checkpoints"
        os.makedirs(self.save_dir, exist_ok=True)

    def encode_state(self, state):
        """Enhanced state encoding"""
        bucket = self.hand_processor.get_bucket(
            state["hole_cards"], 
            state["public_cards"],
            state["street"]
        )
        
        # Normalized features
        pot_odds = state["current_bet"] / (state["pot"] + 1e-8)
        action_history = np.zeros(10)
        action_history[:len(state["action_history"])] = state["action_history"]
        
        features = np.concatenate([
            [bucket / 100.0],
            [pot_odds],
            [state["street"] / 3.0],
            action_history / 100.0,
            [state["current_player"] / 1.0],
            np.array([int(a in state["legal_actions"]) for a in range(3)])
        ])
        
        return torch.FloatTensor(features).to(self.device)

    def _cfr_iteration(self, player):
        state = self.env.reset()
        history = []
        while not state["is_terminal"]:
            encoded_state = self.encode_state(state)
            legal_actions = state["legal_actions"]
            
            # Epsilon-greedy with decay
            if random.random() < self.epsilon:
                action = random.choice(legal_actions)
            else:
                with torch.no_grad():
                    logits, _ = self.strategy_net(encoded_state.unsqueeze(0))
                    action_probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
                    action = np.random.choice(3, p=action_probs)
            
            # Store experience
            history.append((encoded_state, action, player == state["current_player"]))
            
            # Environment step
            next_state, reward, done = self.env.step(action)
            state = next_state

        # Backpropagate values and update networks
        self._process_cfr(history, reward)

    def _process_cfr(self, history, final_reward):
        for encoded_state, action, is_acting_player in reversed(history):
            if is_acting_player:
                state_key = tuple(encoded_state.cpu().numpy().tolist())
                
                # Calculate counterfactual value
                cf_value = final_reward if is_acting_player else -final_reward
                
                # Update cumulative strategy
                with torch.no_grad():
                    _, strategy = self.strategy_net(encoded_state.unsqueeze(0))
                self.cumulative_strategy[state_key] += strategy.squeeze().cpu().numpy()
                
                # Calculate and store regrets
                regret = cf_value - np.dot(self.cumulative_strategy[state_key], cf_value)
                self.cumulative_regrets[state_key][action] += regret
                
                # Store experience
                self.advantage_memory.append((
                    encoded_state.cpu().numpy(),
                    self.cumulative_regrets[state_key].copy()
                ))

    def train(self, iterations=10000, batch_size=512, save_interval=500):
        for _ in range(iterations):
            self.iteration += 1
            
            # CFR iterations
            for player in [0, 1]:
                self._cfr_iteration(player)
            
            # Update networks
            if len(self.advantage_memory) >= batch_size:
                loss = self._update_networks(batch_size)
                self.writer.add_scalar('Loss/Advantage', loss, self.iteration)
            
            # Decay exploration
            self.epsilon *= self.epsilon_decay
            
            # Validation and saving
            if self.iteration % 100 == 0:
                win_rate = self.validate()
                self.writer.add_scalar('Metrics/WinRate', win_rate, self.iteration)
                print(f"Iter {self.iteration}: WinRate={win_rate:.2f}, ε={self.epsilon:.3f}")
            
            if self.iteration % save_interval == 0:
                self.save_checkpoint()

    def _update_networks(self, batch_size):
        batch = random.sample(self.advantage_memory, batch_size)
        states, regrets = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        regrets = torch.FloatTensor(np.array(regrets)).to(self.device)
        
        self.optimizer.zero_grad()
        logits, _ = self.advantage_net(states)
        loss = F.mse_loss(logits, regrets)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def validate(self, num_games=50):
        self.strategy_net.eval()
        wins = 0
        
        with torch.no_grad():
            for _ in range(num_games):
                state = self.env.reset()
                while not state["is_terminal"]:
                    encoded = self.encode_state(state)
                    _, action_probs = self.strategy_net(encoded.unsqueeze(0))
                    action = action_probs.argmax().item()
                    state, reward, _ = self.env.step(action)
                wins += reward > 0
        
        self.strategy_net.train()
        return wins / num_games

    def save_checkpoint(self):
        path = os.path.join(self.save_dir, f"checkpoint_{self.iteration}.pth")
        torch.save({
            'iteration': self.iteration,
            'advantage_net': self.advantage_net.state_dict(),
            'strategy_net': self.strategy_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.advantage_net.load_state_dict(checkpoint['advantage_net'])
        self.strategy_net.load_state_dict(checkpoint['strategy_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.iteration = checkpoint['iteration']
        print(f"Loaded checkpoint from {path} (iter {self.iteration})")

if __name__ == "__main__":
    agent = DeepCFR()
    
    # To resume training:
    # agent.load_checkpoint("checkpoints/checkpoint_1000.pth")
    
    agent.train(iterations=5000, save_interval=500)