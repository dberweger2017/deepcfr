from pettingzoo.classic import texas_holdem_no_limit_v6
from deuces import Evaluator, Card
import numpy as np

class PokerEnv:
    def __init__(self):
        self.env = texas_holdem_no_limit_v6.env()
        self.agents = self.env.possible_agents
        
    def reset(self):
        self.env.reset()
        return self._get_observations()
    
    def step(self, actions):
        self.env.step(actions)
        return self._get_observations()
    
    def _get_observations(self):
        obs_dict = {}
        for agent in self.agents:
            obs, reward, done, trunc, info = self.env.last()
            obs_dict[agent] = {
                'observation': obs['observation'],
                'action_mask': obs['action_mask'],
                'done': done or trunc
            }
        return obs_dict, self.env.rewards
    
    def get_legal_actions(self, agent):
        return np.where(self.env.last()[0]['action_mask'])[0]
    
    def convert_card(self, card_idx):
        """Convert card index to deuces format"""
        if card_idx == 0:
            return None
        rank = (card_idx - 1) // 4
        suit = (card_idx - 1) % 4
        return Card.new(rank * 4 + suit)

class HandProcessor:
    def __init__(self):
        self.evaluator = Evaluator()
    
    def convert_card(self, card_idx):
        if card_idx == 0:
            return None
        rank = (card_idx - 1) // 4
        suit = (card_idx - 1) % 4
        return Card.new(rank * 4 + suit)
    
    def get_strength(self, hole_indices, board_indices):
        hole = [self.convert_card(i) for i in hole_indices if i > 0]
        board = [self.convert_card(i) for i in board_indices if i > 0]
        
        if len(hole) != 2 or len(board) < 3:
            return 0.0
            
        try:
            score = self.evaluator.evaluate(board, hole)
            return 1 - self.evaluator.get_five_card_rank_percentage(score)
        except:
            return 0.0