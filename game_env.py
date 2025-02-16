# game_env.py
import logging
from pettingzoo.classic import texas_holdem_no_limit_v6
from deuces import Evaluator, Card
import numpy as np

logger = logging.getLogger(__name__)

class PokerEnv:
    def __init__(self):
        self.env = texas_holdem_no_limit_v6.env()
        self.agents = self.env.possible_agents
        self.current_round = 0
        self.community_card_indices = []  # Track card indices instead of strings
        self.pot_size = 0
        self.history = []
        
        # Keep existing conversion maps
        self.suit_map = {0: "c", 1: "d", 2: "h", 3: "s"}
        self.rank_map = {0: "2", 1: "3", 2: "4", 3: "5", 4: "6", 5: "7",
                        6: "8", 7: "9", 8: "T", 9: "J", 10: "Q", 11: "K", 12: "A"}

    def reset(self):
        self.env.reset()
        self.current_round = 0
        self.community_cards = []
        self.pot_size = 0
        self.history = []
        return self._get_obs()

    def _convert_card(self, idx):
        """Convert 1-based index to human-readable card (preserved)"""
        if idx == 0: return None
        rank = (idx - 1) // 4
        suit = (idx - 1) % 4
        return f"{self.rank_map[rank]}{self.suit_map[suit]}"

    def _get_player_cards(self, observation):
        """Extract hole cards from observation vector (preserved)"""
        hole_mask = observation[:52]
        return [self._convert_card(i+1) for i in np.where(hole_mask == 1)[0]]

    def _get_community_cards(self, observation):
        try:
            # Get community cards directly from RLCard state
            game = self.env.env.env.game
            public_cards = game.get_public_cards()
            self.community_card_indices = [c.card_id + 1 for c in public_cards]  # Convert to 1-based
            
            # Update current round
            self.current_round = game.round_num
            
            return [self._convert_card(idx) for idx in self.community_card_indices]
        except Exception as e:
            logger.error(f"Community card error: {str(e)}")
            return []

    # In PokerEnv's _get_obs method
    def _get_obs(self):
        observations = {}
        for agent in self.agents:
            obs, reward, termination, truncation, _ = self.env.last()
            done = termination or truncation
            
            # CORRECTED: Pot size is tracked through game state, not chip counts
            self.pot_size = self.env.env.env.game.get_pot_size()  # RLCard-specific method
            # If above doesn't work, track manually:
            # self.pot_size = sum(self.env.env.env.game.get_pot())
            
            observations[agent] = {
                'observation': obs['observation'],
                'action_mask': np.array(obs['action_mask'], dtype=np.float32),
                'done': done
            }
        return observations

    # Preserve all logging functionality
    def _log_game_state(self, agent, action):
        """Full preserved logging with fixed indices"""
        obs = self.env.env.last()[0]['observation']
        
        hole_cards = self._get_player_cards(obs)
        community = self._get_community_cards(obs)
        round_names = ["Pre-flop", "Flop", "Turn", "River"]
        
        logger.debug(f"\n{'='*40}")
        logger.debug(f"Game State (Round {self.current_round+1} - {round_names[self.current_round]})")
        logger.debug(f"Agent: {agent}")
        logger.debug(f"Player 0 Chips: {obs[52]:.1f}")  # Correct index
        logger.debug(f"Player 1 Chips: {obs[53]:.1f}")  # Correct index
        logger.debug(f"Total Pot: {self.pot_size:.1f}")
        logger.debug(f"Community Cards: {community or 'None'}")
        logger.debug(f"Player Cards: {hole_cards}")
        logger.debug(f"Action Mask: {self.env.env.last()[0]['action_mask']}")
        logger.debug(f"{'='*40}\n")

    # Preserve step and close methods
    def step(self, action):
        self.env.step(action)
        self._log_game_state(self.env.env.agent_selection, action)
        logger.debug(f"Current rewards: {self.env.env.rewards}")
        return self._get_obs()

    def close(self):
        self.env.close()

class HandProcessor:
    def __init__(self):
        self.evaluator = Evaluator()
    
    def convert_card(self, idx):
        """Convert 1-based index to deuces Card object"""
        if idx == 0: 
            return None
        
        # Card indices to string mapping
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        suits = ['c', 'd', 'h', 's']  # Clubs, Diamonds, Hearts, Spades
        
        rank_idx = (idx - 1) // 4
        suit_idx = (idx - 1) % 4
        
        try:
            return Card.new(ranks[rank_idx] + suits[suit_idx])
        except IndexError:
            logger.error(f"Invalid card index: {idx}")
            return None
    
    def get_strength(self, hole_indices, board_indices):
        try:
            # Validate input indices
            hole = [self.convert_card(i) for i in hole_indices if i > 0]
            board = [self.convert_card(i) for i in board_indices if i > 0]
            
            logger.debug(f"Hole cards: {hole}")
            logger.debug(f"Board cards: {board}")

            # Handle pre-flop state
            if len(board) < 3:
                return self._estimate_preflop_strength(hole)
                
            score = self.evaluator.evaluate(board, hole)
            return 1 - self.evaluator.get_five_card_rank_percentage(score)
            
        except Exception as e:
            logger.error(f"Hand evaluation failed: {str(e)}")
            return 0.0
        
    def _estimate_preflop_strength(self, hole):
        """Basic pre-flop hand ranking estimation"""
        ranks = sorted([Card.get_rank_int(c) for c in hole])
        suited = Card.get_suit_int(hole[0]) == Card.get_suit_int(hole[1])
        
        # Simple pre-flop hand ranking logic
        if ranks[0] == ranks[1]:
            return 0.9  # Pair
        elif ranks[1] - ranks[0] == 1:
            return 0.7 if suited else 0.6  # Suited/unsuited connector
        elif suited:
            return 0.5  # Suited
        return 0.4  # Unconnected/unsuited