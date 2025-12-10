# Assignment: Reinforcement Learning

This assignment is part of the **Machine Learning course at POSTECH**, focusing on implementing classic **Dynamic Programming** algorithms for **Reinforcement Learning** (RL) in a GridWorld environment.

## Overview

The goal of this assignment is to understand and implement **Value Iteration** and **Policy Iteration** to solve Markov Decision Processes (MDPs).
You will work with a randomly generated Maze environment.

### Covered Topics

- **Markov Decision Processes (MDPs):** States, Actions, Rewards, Transitions.
- **Dynamic Programming:**
  - Value Iteration
  - Policy Iteration

## The Maze Environment

The environment is a procedurally generated maze (using the Aldous-Broder algorithm).

- **States:** Grid cells.
- **Actions:** UP, DOWN, LEFT, RIGHT.
- **Rewards:**
  - Living penalty (reward = -1) at most states.
  - Goal reward (reward = 10) at a specific state.
- **Transitions:** Deterministic.
  - Moving into a wall keeps you in the same state.
  - Moving into an open space updates your state.
- **Terminal States**: None.
  - There are no terminal states in this environment (infinite horizon setup).

The `Maze` class is provided in `rl/__init__.py`. It handles generation and state management.

## Tasks Breakdown

*Note on Update Mechanism: The framework is designed for Synchronous Updates (not in-place). The necessary infrastructure (i.e., copying of value functions) is already implemented in the skeleton code.*

### 1. Value Iteration (`value_iteration`)

Implement the Value Iteration algorithm.Your main task is to implement the Bellman Optimality Equation inside the main loop.

Implementation Requirements:

- Bellman Optimality Update: Inside the loop, for every state $s$, calculate the value of all possible actions and update $V(s)$ with the maximum value.

$$
V_{new}(s) \leftarrow \max_{a} \left( R(s) + \gamma V_{old}(\text{next}(s, a)) \right)
$$

*Note: Since the environment is deterministic, the summation over $s'$ is removed. Also, we use the state-reward definition $R(s)$ (reward at current state) rather than the transition-reward $R(s, a, s')$.*
<!-- 2. Synchronous Update:Ensure you use the values from the previous iteration (V) to compute the values for the current iteration (new_V).
3. Convergence Check:The loop should terminate when the maximum change in value function ($\Delta$) is less than $\theta$. -->

### 2. Policy Iteration (`policy_iteration`)

Implement the Policy Iteration algorithm.

This requires implementing two distinct steps inside the main loop:

Implementation Requirements:

1. **Step 1 (Policy Evaluation)**: Implement the Bellman Expectation Equation to evaluate the current policy $\pi$.
    - Update $V(s)$ based on the action selected by the current policy:
    $$
    V_{new}(s) \leftarrow R(s) + \gamma V_{old}(\text{next}(s, \pi(s)))
    $$
    - Repeat this step until $V$ converges (inner loop).
2. **Step 2 (Policy Improvement)**: Implement the Greedy Update to improve the policy.
    - For each state, calculate the Q-value for all actions and update the policy to choose the action that maximizes the Q-value.
    $$
    \pi(s) \leftarrow \arg\max_a \left( R(s) + \gamma V(\text{next}(s, a)) \right)
    $$
    - Check if the policy has changed to determine if the main loop should terminate.

*Note: Since the environment is deterministic, the summation over $s'$ is removed. Also, we use the state-reward definition $R(s)$ (reward at current state) rather than the transition-reward $R(s, a, s')$.*

## Submission

1. Implement code in [`rl/__init__.py`](rl/__init__.py).
2. Validate your implementation by running the tests in the `tests` directory using pytest.
   - Please ensure all public tests pass before submitting your assignment.
3. Submit only [`rl/__init__.py`](rl/__init__.py) to the course platform.

## Visualizing Results

You can visualize your implementation's results using the provided scripts.

- **Animation (`animation.py`)**: Generates GIF animations showing the learning process.

    ```bash
    uv run rl/animation.py
    ```

    This will create `value_iteration.gif` and `policy_iteration.gif`.

  - **Value Iteration Animation**: Shows the value function converging over iterations.
  - **Policy Iteration Animation**: Shows the policy and value function improvement steps.

*Note: You can check example outputs in the rl/visuals/ directory.*

<!-- *Note: These scripts require `matplotlib`. It is included in the project dependencies.*
*To install dependencies, you can use `uv sync` or `pip install -e .`.* -->

## Credits

- Based on `mazemdp` by Sally Gao, Duncan Rule, Yi Hao.
- Sutton & Barto, Reinforcement Learning: An Introduction.
