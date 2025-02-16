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
    D --> E[Forward Pass through PokerNet via CFRNetwork]
    E --> F{Agent Type?}
    F -- Training Agent --> G[Select Action with Epsilon-Greedy]
    F -- Opponent Agent --> H[Select Action using Network Policy]
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

## Key Decisions

- **Simplicity over Complexity**:  
  The project uses a straightforward two-layer network architecture, which I think strikes a good balance between performance and interpretability.

- **Separation of Concerns**:  
  By splitting the environment, model definitions, and training logic into separate files, the code remains modular and easier to manage.

- **Detailed Logging**:  
  Extra logging is built into both the environment and the training process. This isn't just for debugging; it's essential for understanding learning dynamics in a complex domain like poker.

- **Epsilon-Greedy Exploration**:  
  The decision to decay epsilon ensures thorough exploration in the early stages of training, then gradually shifts towards exploitation as the models improve. I find this approach both practical and effective.

## Running the Project

Ensure you have the necessary dependencies installed (e.g., PyTorch, PettingZoo, and Deuces). Then, simply run:

```bash
python deep_cfr.py --log INFO
```

This command starts the training process. Adjust parameters like the number of iterations, batch size, or save interval as needed.

## Final Thoughts

I'm pretty happy with the decisions made here. The project isn't over-engineered; it's a focused exploration of deep learning applied to poker using CFR. If you're into game theory or want to see how deep learning can work with complex decision-making processes, give this project a try!