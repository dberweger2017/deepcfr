# Deep CFR Poker Project

This project implements a Deep Counterfactual Regret (CFR) approach for training poker agents. It’s not overhyped—just a solid, experimental setup to explore deep learning for poker strategy.

## Overview

The project is split into three main parts:
- **models.py**: Contains the neural network architectures for evaluating poker states.
- **game_env.py**: Wraps around the PettingZoo Texas Hold'em environment with extra logging and card conversion utilities.
- **deep_cfr.py**: Implements the Deep CFR algorithm, handling training iterations, regret updates, and network updates.

## What is CFR?

Counterfactual Regret Minimization (CFR) is a technique used to compute approximate Nash equilibria in imperfect information games (like poker). Instead of trying to optimize a direct policy, CFR focuses on minimizing regret for actions not taken, which gradually refines the strategy over time.

## Training Process

1. **Simulation & State Encoding**:  
   The environment is reset and each agent’s state is encoded. This involves converting raw observations into features like hand strength and pot size.

2. **Action Selection**:  
   For the training agent, the system uses an epsilon-greedy approach: with probability `epsilon`, it explores randomly; otherwise, it follows the network's suggested strategy. The opponent uses its own network to pick actions.

3. **Regret & Strategy Update**:  
   After each iteration, the algorithm calculates counterfactual values and immediate regrets, updating cumulative regrets and strategies accordingly.

4. **Network Updates**:  
   Although the current demo disables network updates, there are placeholders for training the advantage and strategy networks using MSE and KL-divergence losses respectively.

5. **Checkpointing**:  
   Models are periodically saved, ensuring you can resume training or analyze progress.

## Key Decisions

- **Simplicity over Complexity**:  
  The project uses a straightforward two-layer network architecture, which I believe strikes a good balance between performance and interpretability.

- **Separation of Concerns**:  
  By splitting the environment, model definitions, and training logic into separate files, the code stays modular and easier to manage.

- **Detailed Logging**:  
  Extra logging is built into the environment and training process. This isn’t just for debugging—it’s essential for understanding the learning dynamics in a complex domain like poker.

- **Epsilon-Greedy Exploration**:  
  The decision to decay epsilon ensures exploration initially and gradual exploitation as training progresses. I think it’s a sensible choice given the uncertainties in poker.

## Running the Project

Make sure you have the necessary dependencies installed (e.g., PyTorch, PettingZoo, and Deuces). Then, simply run:

```bash
python deep_cfr.py --log INFO
```