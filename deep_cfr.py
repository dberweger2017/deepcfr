# deep_cfr.py
import torch
import torch.nn.functional as F
import numpy as np
import random
import os
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from models import NLHEPolicyNetwork
from game_env import NLHEEnvironment, HandProcessor

class DeepCFR:
    def __init__(self):
        self.env = NLHEEnvironment()
        self.hand_processor = HandProcessor()
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Our state encoder now outputs 17 features.
        self.state_dim = 17
        
        # Initialize networks with the correct input dimension.
        self.advantage_net = NLHEPolicyNetwork(state_dim=self.state_dim).to(self.device)
        self.strategy_net = NLHEPolicyNetwork(state_dim=self.state_dim).to(self.device)
        
        # Separate optimizers for the two networks
        self.adv_optimizer = torch.optim.Adam(self.advantage_net.parameters(), lr=1e-4)
        self.strat_optimizer = torch.optim.Adam(self.strategy_net.parameters(), lr=5e-5)
        
        # Training state
        self.epsilon = 0.1
        self.epsilon_decay = 0.999
        self.iteration = 0
        self.advantage_memory = deque(maxlen=100000)
        self.strategy_memory = deque(maxlen=100000)

        # Logging
        self.writer = SummaryWriter()
        self.save_dir = "checkpoints"
        os.makedirs(self.save_dir, exist_ok=True)

    def encode_state(self, state):
        """Enhanced state encoding producing 17 features."""
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
            [bucket / 100.0],                    # 1 feature
            [pot_odds],                          # 1 feature
            [state["street"] / 3.0],               # 1 feature
            action_history / 100.0,                # 10 features
            [state["current_player"] / 1.0],       # 1 feature
            np.array([int(a in state["legal_actions"]) for a in range(3)])  # 3 features
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
            
            # If it's the acting player's turn, store the current strategy.
            if player == state["current_player"]:
                with torch.no_grad():
                    logits, _ = self.strategy_net(encoded_state.unsqueeze(0))
                    strategy = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
                self.strategy_memory.append((encoded_state.cpu().numpy(), strategy))
            
            # Record the experience (we'll use final_reward later)
            history.append((encoded_state, action, player == state["current_player"]))
            
            # Environment step
            next_state, reward, done = self.env.step(action)
            state = next_state

        # After the game ends, process the trajectory for advantage updates.
        self._process_cfr(history, reward)

    def _process_cfr(self, history, final_reward):
        """
        For each state where the acting player decided,
        we set the target advantage for the chosen action as final_reward.
        """
        for encoded_state, action, is_acting_player in reversed(history):
            if is_acting_player:
                target_advantage = np.zeros(3)
                target_advantage[action] = final_reward
                self.advantage_memory.append((
                    encoded_state.cpu().numpy(),
                    target_advantage
                ))

    def _update_advantage_network(self, batch_size):
        batch = random.sample(self.advantage_memory, batch_size)
        states, targets = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        targets = torch.FloatTensor(np.array(targets)).to(self.device)
        
        self.adv_optimizer.zero_grad()
        # Use only the action head output as our predicted advantage.
        logits, _ = self.advantage_net(states)
        loss = F.mse_loss(logits, targets)
        loss.backward()
        self.adv_optimizer.step()
        
        return loss.item()

    def _update_strategy_network(self, batch_size):
        batch = random.sample(self.strategy_memory, batch_size)
        states, target_strats = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        target_strats = torch.FloatTensor(np.array(target_strats)).to(self.device)
        
        self.strat_optimizer.zero_grad()
        pred_logits, _ = self.strategy_net(states)
        log_probs = F.log_softmax(pred_logits, dim=-1)
        loss = F.kl_div(log_probs, target_strats, reduction='batchmean')
        loss.backward()
        self.strat_optimizer.step()
        
        return loss.item()

    def train(self, iterations=10000, batch_size=512, save_interval=500):
        for _ in range(iterations):
            self.iteration += 1
            
            # Run a CFR iteration for both players.
            for player in [0, 1]:
                self._cfr_iteration(player)
            
            # Update advantage network if we have enough samples.
            if len(self.advantage_memory) >= batch_size:
                adv_loss = self._update_advantage_network(batch_size)
                self.writer.add_scalar('Loss/Advantage', adv_loss, self.iteration)
            
            # Update strategy network if we have enough samples.
            if len(self.strategy_memory) >= batch_size:
                strat_loss = self._update_strategy_network(batch_size)
                self.writer.add_scalar('Loss/Strategy', strat_loss, self.iteration)
            
            # Decay exploration
            self.epsilon *= self.epsilon_decay
            
            # Validation and saving
            if self.iteration % 100 == 0:
                win_rate = self.validate()
                self.writer.add_scalar('Metrics/WinRate', win_rate, self.iteration)
                print(f"Iter {self.iteration}: WinRate={win_rate:.2f}, ε={self.epsilon:.3f}")
            
            if self.iteration % save_interval == 0:
                self.save_checkpoint()

    def validate(self, num_games=50):
        self.strategy_net.eval()
        wins = 0
        
        with torch.no_grad():
            for _ in range(num_games):
                state = self.env.reset()
                while not state["is_terminal"]:
                    encoded = self.encode_state(state)
                    logits, _ = self.strategy_net(encoded.unsqueeze(0))
                    action = logits.argmax(dim=-1).item()
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
            'adv_optimizer': self.adv_optimizer.state_dict(),
            'strat_optimizer': self.strat_optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.advantage_net.load_state_dict(checkpoint['advantage_net'])
        self.strategy_net.load_state_dict(checkpoint['strategy_net'])
        self.adv_optimizer.load_state_dict(checkpoint['adv_optimizer'])
        self.strat_optimizer.load_state_dict(checkpoint['strat_optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.iteration = checkpoint['iteration']
        print(f"Loaded checkpoint from {path} (iter {self.iteration})")

if __name__ == "__main__":
    agent = DeepCFR()
    
    # To resume training:
    # agent.load_checkpoint("checkpoints/checkpoint_1000.pth")
    
    agent.train(iterations=5000, save_interval=500)