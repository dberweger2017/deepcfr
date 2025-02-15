# game_env.py
from deuces import Evaluator, Deck, Card
import open_spiel.python.rl_environment as rl_env
import numpy as np

class NLHEEnvironment:
    def __init__(self):
        self.env = rl_env.Environment("no_limit_texas_holdem", num_players=2)
        self.evaluator = Evaluator()
        
        # Proper OpenSpiel action mapping
        self.ACTION_MEANING = {
            0: "FOLD",
            1: "CALL",
            2: "RAISE"
        }

    def reset(self):
        time_step = self.env.reset()
        return self._extract_state(time_step)
    
    def step(self, action):
        # Convert discrete action to OpenSpiel's expected format
        try:
            action = int(action)
            if action not in self.env.action_spec()["actions"]:
                action = 1  # Default to CALL
        except:
            action = 1
            
        time_step = self.env.step([action])
        return self._extract_state(time_step), time_step.rewards[0], time_step.last()

    def _extract_state(self, time_step):
        obs = time_step.observations["info_state"][0]
        return {
            "legal_actions": list(time_step.observations["legal_actions"][0]),
            "hole_cards": obs[:2],
            "public_cards": obs[2:7],
            "pot": obs[7],
            "stacks": obs[8:10],
            "current_player": time_step.current_player(),
            "street": int(obs[9]),
            "is_terminal": time_step.last()
        }

class HandProcessor:
    def __init__(self):
        self.evaluator = Evaluator()
        
    def get_bucket(self, hole_indices, board_indices, street):
        # Convert OpenSpiel card indices to deuces cards
        hole_cards = [Card.new(int(i)) for i in hole_indices if i != 0]
        board_cards = [Card.new(int(i)) for i in board_indices if i != 0]
        
        if street == 0:
            return self._preflop_bucket(hole_cards)
        return self._postflop_equity(hole_cards, board_cards)
    
    def _preflop_bucket(self, hole_cards):
        if len(hole_cards) != 2:
            return 0
            
        ranks = sorted([Card.get_rank_int(c) for c in hole_cards], reverse=True)
        suited = Card.get_suit_int(hole_cards[0]) == Card.get_suit_int(hole_cards[1])
        return hash(f"{ranks[0]}-{ranks[1]}-{suited}") % 100
    
    def _postflop_equity(self, hole_cards, board_cards):
        if len(board_cards) < 3 or len(hole_cards) != 2:
            return 0
            
        try:
            score = self.evaluator.evaluate(board_cards, hole_cards)
            return int((1 - self.evaluator.get_five_card_rank_percentage(score)) * 100)
        except:
            return 0
        
    def _convert_card(self, openspiel_idx):
        rank = openspiel_idx // 4  # 0-12 (2-A)
        suit = openspiel_idx % 4    # 0=♣,1=♦,2=♥,3=♠
        return Card.new(rank * 4 + suit)