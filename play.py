import pokers as pkrs
import torch
import numpy as np
import argparse
import os
import random
from deep_cfr import DeepCFRAgent
from model import set_verbose

def get_action_description(action):
    """Convert a pokers action to a human-readable string."""
    if action.action == pkrs.ActionEnum.Fold:
        return "Fold"
    elif action.action == pkrs.ActionEnum.Check:
        return "Check"
    elif action.action == pkrs.ActionEnum.Call:
        return "Call"
    elif action.action == pkrs.ActionEnum.Raise:
        return f"Raise to {action.amount:.2f}"
    else:
        return f"Unknown action: {action.action}"

def card_to_string(card):
    """Convert a poker card to a readable string."""
    suits = {0: "♣", 1: "♦", 2: "♥", 3: "♠"}
    ranks = {0: "2", 1: "3", 2: "4", 3: "5", 4: "6", 5: "7", 6: "8", 
             7: "9", 8: "10", 9: "J", 10: "Q", 11: "K", 12: "A"}
    
    return f"{ranks[int(card.rank)]}{suits[int(card.suit)]}"

def display_game_state(state, player_id=0):
    """Display the current game state in a human-readable format."""
    print("\n" + "="*70)
    print(f"Stage: {state.stage.name}")
    print(f"Pot: ${state.pot:.2f}")
    print(f"Button position: Player {state.button}")
    
    # Show community cards
    community_cards = " ".join([card_to_string(card) for card in state.public_cards])
    print(f"Community cards: {community_cards if community_cards else 'None'}")
    
    # Show player's hand
    hand = " ".join([card_to_string(card) for card in state.players_state[player_id].hand])
    print(f"Your hand: {hand}")
    
    # Show all players' states
    print("\nPlayers:")
    for i, p in enumerate(state.players_state):
        status = "YOU" if i == player_id else "AI"
        active = "Active" if p.active else "Folded"
        print(f"Player {i} ({status}): ${p.stake:.2f} - Bet: ${p.bet_chips:.2f} - {active}")
    
    # Show legal actions for human player if it's their turn
    if state.current_player == player_id:
        print("\nLegal actions:")
        for action_enum in state.legal_actions:
            if action_enum == pkrs.ActionEnum.Fold:
                print("  f: Fold")
            elif action_enum == pkrs.ActionEnum.Check:
                print("  c: Check")
            elif action_enum == pkrs.ActionEnum.Call:
                # Calculate call amount
                call_amount = max(0, state.min_bet - state.players_state[player_id].bet_chips)
                print(f"  c: Call ${call_amount:.2f}")
            elif action_enum == pkrs.ActionEnum.Raise:
                min_raise = state.min_bet
                max_raise = state.players_state[player_id].stake
                print(f"  r: Raise (min: ${min_raise:.2f}, max: ${max_raise:.2f})")
                print("    h: Raise half pot")
                print("    p: Raise pot")
                print("    m: Custom raise amount")
    
    print("="*70)

def get_human_action(state, player_id=0):
    """Get action from human player via console input."""
    while True:
        action_input = input("Your action (f=fold, c=check/call, r=raise, h=half pot, p=pot, m=custom): ").strip().lower()
        
        # Process fold
        if action_input == 'f' and pkrs.ActionEnum.Fold in state.legal_actions:
            return pkrs.Action(pkrs.ActionEnum.Fold)
        
        # Process check/call
        elif action_input == 'c':
            if pkrs.ActionEnum.Check in state.legal_actions:
                return pkrs.Action(pkrs.ActionEnum.Check)
            elif pkrs.ActionEnum.Call in state.legal_actions:
                return pkrs.Action(pkrs.ActionEnum.Call)
        
        # Process raise shortcuts
        elif action_input in ['r', 'h', 'p', 'm'] and pkrs.ActionEnum.Raise in state.legal_actions:
            player_state = state.players_state[player_id]
            min_bet = state.min_bet
            max_bet = player_state.stake
            
            if action_input == 'h':  # Half pot
                bet_amount = min(state.pot * 0.5, max_bet)
                return pkrs.Action(pkrs.ActionEnum.Raise, bet_amount)
            
            elif action_input == 'p':  # Full pot
                bet_amount = min(state.pot, max_bet)
                return pkrs.Action(pkrs.ActionEnum.Raise, bet_amount)
            
            elif action_input == 'm' or action_input == 'r':  # Custom amount
                while True:
                    try:
                        amount_str = input(f"Enter raise amount (min: {min_bet:.2f}, max: {max_bet:.2f}): ")
                        amount = float(amount_str)
                        if min_bet <= amount <= max_bet:
                            return pkrs.Action(pkrs.ActionEnum.Raise, amount)
                        else:
                            print(f"Amount must be between {min_bet:.2f} and {max_bet:.2f}")
                    except ValueError:
                        print("Please enter a valid number")
        
        print("Invalid action. Please try again.")

def play_against_models(model_paths, player_position=0, initial_stake=200.0, small_blind=1.0, big_blind=2.0, verbose=False):
    """
    Play against a set of AI models.
    
    Args:
        model_paths: List of paths to model checkpoint files
        player_position: Position of the human player (0-5)
        initial_stake: Starting chip count for all players
        small_blind: Small blind amount
        big_blind: Big blind amount
        verbose: Whether to show detailed output
    """
    set_verbose(verbose)
    
    # Check if we have enough models
    if len(model_paths) < 5:
        print("Warning: Not enough models provided. Using random agents to fill remaining positions.")
    
    # Load models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agents = []
    
    # Create agents for each position
    for i in range(6):
        if i == player_position:
            # Human player
            agents.append(None)
        else:
            # Fill with models or random agents
            model_idx = (i - 1) if i > player_position else i
            if model_idx < len(model_paths) and model_paths[model_idx] is not None:
                # Load model
                try:
                    agent = DeepCFRAgent(player_id=i, num_players=6, device=device)
                    agent.load_model(model_paths[model_idx])
                    agents.append(agent)
                    print(f"Loaded model for position {i}: {os.path.basename(model_paths[model_idx])}")
                except Exception as e:
                    print(f"Error loading model for position {i}: {e}")
                    print("Using random agent instead")
                    agents.append(RandomAgent(i))
            else:
                # Use random agent
                agents.append(RandomAgent(i))
                print(f"Using random agent for position {i}")
    
    # Track game statistics
    num_games = 0
    total_profit = 0
    player_stake = initial_stake
    
    # Main game loop
    while True:
        if player_stake <= 0:
            print("\nYou're out of chips! Game over.")
            break
        
        # Ask if player wants to continue
        if num_games > 0:
            choice = input("\nContinue playing? (y/n): ").strip().lower()
            if choice != 'y':
                print("Thanks for playing!")
                break
        
        num_games += 1
        print(f"\n--- Game {num_games} ---")
        print(f"Your current balance: ${player_stake:.2f}")
        
        # Rotate button position for fairness
        button_pos = (num_games - 1) % 6
        
        # Create a new poker game
        state = pkrs.State.from_seed(
            n_players=6,
            button=button_pos,
            sb=small_blind,
            bb=big_blind,
            stake=initial_stake,
            seed=random.randint(0, 10000)
        )
        
        # Play until the game is over
        while not state.final_state:
            current_player = state.current_player
            
            # Display game state before human acts
            if current_player == player_position:
                display_game_state(state, player_position)
                action = get_human_action(state, player_position)
                print(f"You chose: {get_action_description(action)}")
            else:
                # Abbreviated state display for AI turns
                print(f"\nPlayer {current_player}'s turn")
                action = agents[current_player].choose_action(state)
                print(f"Player {current_player} chose: {get_action_description(action)}")
            
            # Apply the action
            state = state.apply_action(action)
        
        # Game is over, show results
        print("\n--- Game Over ---")
        
        # Show all players' hands
        print("Final hands:")
        for i, p in enumerate(state.players_state):
            if p.active:
                hand = " ".join([card_to_string(card) for card in p.hand])
                print(f"Player {i}: {hand}")
        
        # Show community cards
        community_cards = " ".join([card_to_string(card) for card in state.public_cards])
        print(f"Community cards: {community_cards}")
        
        # Show results
        print("\nResults:")
        for i, p in enumerate(state.players_state):
            player_type = "YOU" if i == player_position else "AI"
            print(f"Player {i} ({player_type}): ${p.reward:.2f}")
        
        # Update player's stake
        game_profit = state.players_state[player_position].reward
        total_profit += game_profit
        player_stake += game_profit
        
        print(f"\nThis game: {'Won' if game_profit > 0 else 'Lost'} ${abs(game_profit):.2f}")
        print(f"Running total: ${total_profit:.2f}")
        print(f"Current balance: ${player_stake:.2f}")
    
    # Show overall statistics
    print("\n--- Overall Statistics ---")
    print(f"Games played: {num_games}")
    print(f"Total profit: ${total_profit:.2f}")
    print(f"Average profit per game: ${total_profit/num_games if num_games > 0 else 0:.2f}")
    print(f"Final balance: ${player_stake:.2f}")

class RandomAgent:
    """Simple random agent for poker."""
    def __init__(self, player_id):
        self.player_id = player_id
        
    def choose_action(self, state):
        """Choose a random legal action."""
        if not state.legal_actions:
            # Default action if no legal actions (shouldn't happen)
            return pkrs.Action(pkrs.ActionEnum.Call)
        
        # Select a random legal action
        action_enum = random.choice(state.legal_actions)
        
        # For raises, select a random amount between min and max
        if action_enum == pkrs.ActionEnum.Raise:
            player_state = state.players_state[state.current_player]
            min_amount = state.min_bet
            max_amount = player_state.stake  # All-in
            
            # Choose between 0.5x pot, 1x pot, or a random amount
            pot_amounts = [state.pot * 0.5, state.pot]
            valid_amounts = [amt for amt in pot_amounts if min_amount <= amt <= max_amount]
            
            if valid_amounts:
                amount = random.choice(valid_amounts)
            else:
                amount = random.uniform(min_amount, max_amount)
                
            return pkrs.Action(action_enum, amount)
        else:
            return pkrs.Action(action_enum)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play poker against AI models')
    parser.add_argument('--models', nargs='+', default=[], help='Paths to model checkpoint files')
    parser.add_argument('--position', type=int, default=0, help='Your position at the table (0-5)')
    parser.add_argument('--stake', type=float, default=200.0, help='Initial stake')
    parser.add_argument('--sb', type=float, default=1.0, help='Small blind amount')
    parser.add_argument('--bb', type=float, default=2.0, help='Big blind amount')
    parser.add_argument('--verbose', action='store_true', help='Show detailed output')
    args = parser.parse_args()
    
    # Fill with None if not enough models provided
    model_paths = args.models + [None] * (5 - len(args.models))
    
    # Start the game
    play_against_models(
        model_paths=model_paths,
        player_position=args.position,
        initial_stake=args.stake,
        small_blind=args.sb,
        big_blind=args.bb,
        verbose=args.verbose
    )