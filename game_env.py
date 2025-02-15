from pettingzoo.classic import texas_holdem_no_limit_v6
from deuces import Evaluator, Card
import numpy as np

class PokerEnv:
    def __init__(self):
        self.env = texas_holdem_no_limit_v6.env()
        self.agents = self.env.possible_agents
        self.dones = {agent: False for agent in self.agents}

    def reset(self):
        self.env.reset()
        self.dones = {agent: False for agent in self.agents}
        return self._get_obs()
    
    def step(self, action):
        self.env.step(action)
        return self._get_obs()
    
    def _get_obs(self):
        observations = {}
        print("\nGetting observations...")
        for agent in self.agents:
            last_output = self.env.last()
            print(f"Agent: {agent} | Last output type: {type(last_output)}")
            
            obs_dict, reward, termination, truncation, _ = last_output
            done = termination or truncation
            
            print(f"Observation keys: {obs_dict.keys()}")
            print(f"Observation sample: {obs_dict['observation'][:10]}...")
            print(f"Action mask: {obs_dict['action_mask']}")
            
            observations[agent] = {
                'observation': obs_dict['observation'],
                'action_mask': np.array(obs_dict['action_mask'], dtype=np.float32),
                'done': done
            }
        return observations
    
    def close(self):
        self.env.close()

class HandProcessor:
    def __init__(self):
        self.evaluator = Evaluator()
    
    def convert_card(self, idx):
        """Convert 1-based index to deuces Card object"""
        if idx == 0: return None
        rank = (idx - 1) // 4
        suit = (idx - 1) % 4
        return Card.new(rank * 4 + suit)
    
    def get_strength(self, hole_indices, board_indices):
        try:
            hole = [self.convert_card(i) for i in hole_indices if i > 0]
            board = [self.convert_card(i) for i in board_indices if i > 0]
            
            if len(hole) != 2 or len(board) < 3:
                return 0.0
                
            score = self.evaluator.evaluate(board, hole)
            return 1 - self.evaluator.get_five_card_rank_percentage(score)
        except:
            return 0.0