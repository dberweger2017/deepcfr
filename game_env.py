from pettingzoo.classic import texas_holdem_no_limit_v6
from deuces import Evaluator, Card
import numpy as np

class PokerEnv:
    def __init__(self):
        self.env = texas_holdem_no_limit_v6.env()
        self.agents = self.env.possible_agents
        self.current_round = 0
        self.community_cards = []
        self.pot_size = 0
        self.history = []
        
        # Card conversion maps
        self.suit_map = {0: "♠", 1: "♥", 2: "♦", 3: "♣"}
        self.rank_map = {
            0: "2", 1: "3", 2: "4", 3: "5", 4: "6", 5: "7",
            6: "8", 7: "9", 8: "T", 9: "J", 10: "Q", 11: "K", 12: "A"
        }

    def reset(self):
        self.env.reset()
        self.current_round = 0
        self.community_cards = []
        self.pot_size = 0
        self.history = []
        return self._get_obs()

    def _convert_card(self, idx):
        """Convert 1-based index to human-readable card"""
        if idx == 0: return None
        rank = (idx - 1) // 4
        suit = (idx - 1) % 4
        return f"{self.rank_map[rank]}{self.suit_map[suit]}"

    def _get_player_cards(self, observation):
        """Extract hole cards from observation vector"""
        hole_mask = observation[:52]
        return [self._convert_card(i+1) for i in np.where(hole_mask == 1)[0]]

    def _get_community_cards(self, observation):
        """Extract community cards from observation vector"""
        board_mask = observation[52:52+52*5]
        new_cards = [self._convert_card(i+1) for i in np.where(board_mask == 1)[0]]
        
        # Detect new community cards
        if len(new_cards) > len(self.community_cards):
            self.community_cards = new_cards
            if len(new_cards) == 3:
                self.current_round = 1  # Flop
            elif len(new_cards) == 4:
                self.current_round = 2  # Turn
            elif len(new_cards) == 5:
                self.current_round = 3  # River

        return self.community_cards

    def _get_obs(self):
        observations = {}
        for agent in self.agents:
            obs, reward, termination, truncation, _ = self.env.last()
            done = termination or truncation
            
            # Track pot size
            self.pot_size = obs['observation'][52] + obs['observation'][53]
            
            observations[agent] = {
                'observation': obs['observation'],
                'action_mask': np.array(obs['action_mask'], dtype=np.float32),
                'done': done
            }
        return observations

    def _log_game_state(self, agent, action):
        """Detailed game state logging"""
        obs = self.env.env.last()[0]['observation']
        
        # Get card information
        hole_cards = self._get_player_cards(obs)
        community = self._get_community_cards(obs)
        round_names = ["Pre-flop", "Flop", "Turn", "River"]
        
        # Action translation
        action_names = {
            0: "Fold", 1: "Call", 2: "Raise Half Pot",
            3: "Raise Full Pot", 4: "All-In"
        }
        
        print(f"\n{'='*40}")
        print(f"Game State (Round {self.current_round+1} - {round_names[self.current_round]})")
        print(f"Agent: {agent}")
        print(f"Player 0 Chips: {obs['observation'][52]:.1f}")
        print(f"Player 1 Chips: {obs['observation'][53]:.1f}")
        print(f"Total Pot: {self.pot_size:.1f}")
        print(f"Community Cards: {community or 'None'}")
        print(f"Player Cards: {hole_cards}")
        print(f"Action Taken: {action_names.get(action, 'Unknown')} ({action})")
        print(f"Action Mask: {self.env.env.last()[0]['action_mask']}")
        print(f"{'='*40}\n")

    def step(self, action):
        self.env.step(action)
        self._log_game_state(self.env.env.agent_selection, action)
        return self._get_obs()

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