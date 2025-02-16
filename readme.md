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
   The environment is reset and each agent’s state is encoded using the `HandProcessor`. This step converts raw observations into features like hand strength and pot size.

2. **Action Selection**:  
   - The **training agent** uses an epsilon-greedy approach: with probability `epsilon`, it explores randomly; otherwise, it follows the network's suggested strategy.
   - The **opponent** selects actions based on its own network's policy.

3. **Regret & Strategy Update**:  
   After each iteration, the algorithm calculates counterfactual values and immediate regrets. It updates the cumulative regrets and strategy counts accordingly.

4. **Network Updates**:  
   Although the current demo disables network updates, there are placeholders for training:
   - The **Advantage Network** is updated using Mean Squared Error (MSE) loss.
   - The **Strategy Network** is updated using Kullback-Leibler (KL) divergence loss.
   - There is also a soft target update mechanism.

5. **Checkpointing & Exploration Decay**:  
   Models are periodically saved, and the epsilon value decays over time to shift from exploration to exploitation.

## Training Process Diagram

Below is a Mermaid diagram outlining the training process with an emphasis on the models and CFR components. Note that while GitHub now supports Mermaid (in beta), some Markdown renderers might not render it.

```mermaid
flowchart TD
    A[Start Training Process] --> B[Reset Poker Environment]
    B --> C[Encode State using HandProcessor]
    C --> D[Obtain Features: Hand Strength, Pot Size, etc.]
    D --> E[Forward Pass through PokerNet (via CFRNetwork)]
    E --> F{Agent Type?}
    F -- Training Agent --> G[Select Action (Epsilon-Greedy)]
    F -- Opponent Agent --> H[Select Action (Network Policy)]
    G --> I[CFR Iteration: Record State, Action, Reach Probabilities]
    H --> I
    I --> J[Compute Immediate Regrets (CFR Core)]
    J --> K[Update Cumulative Regrets & Strategy Counts]
    K --> L[Store (State, Regrets) in Advantage Memory]
    L --> M[Update Advantage Network (MSE Loss)]
    K --> N[Update Strategy Network (KL Loss)]
    M --> O[Soft Update Target Networks]
    N --> O
    O --> P[Decay Epsilon & Checkpoint Models]
    P --> Q[Next Iteration]
    Q --> C
```