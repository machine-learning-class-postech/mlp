"""
Maze MDP Environment and Solvers.

This module provides a grid-world Maze environment and DP algorithms
(Value Iteration, Policy Iteration) to solve it.

Attribution:
    The Maze environment logic is based on `mazemdp` by Sally Gao, Duncan Rule, and Yi Hao.
"""

from __future__ import annotations

import numpy as np

# Constants for Actions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]
ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]

# Maze Constants
WALL = 1
EMPTY = 0


class Maze:
    def __init__(self, h: int, w: int, seed: int | None = None):
        """
        Defines a grid-based Maze environment for Reinforcement Learning.

        The maze assumes a grid structure where cells can be empty (navigable) or walls (blocked).
        The agent receives a reward for reaching the goal and a small penalty for each step
        to encourage the shortest path.

        Args:
            h: Height of the internal grid structure (actual height will be 2*h + 1).
            w: Width of the internal grid structure (actual width will be 2*w + 1).
            seed: Random seed for reproducibility.
        """
        self.h = h
        self.w = w
        self.H = (2 * h) + 1
        self.W = (2 * w) + 1
        self.rng = np.random.default_rng(seed)

        # Grid: 1 = Wall, 0 = Empty
        self.grid = np.ones((self.H, self.W), dtype=np.int32)
        self._generate()

        # Reward map: -1 everywhere, 10 at goal (1, 1)
        # Note: mazemdp places goal at (1,1) with reward 10.
        self.rewards = np.full((self.H, self.W), -1.0, dtype=np.float64)
        self.rewards[1, 1] = 10.0

    def _generate(self) -> None:
        """Generates the maze using Aldous-Broder algorithm."""
        # Start at a random odd coordinate
        crow = self.rng.integers(1, self.h + 1) * 2 - 1
        ccol = self.rng.integers(1, self.w + 1) * 2 - 1
        self.grid[crow, ccol] = EMPTY

        num_visited = 1
        total_cells = self.h * self.w

        while num_visited < total_cells:
            # Find neighbors
            neighbors: list[tuple[np.int64, np.int64]] = []
            # Check up (r-2)
            if crow > 1:
                neighbors.append((crow - 2, ccol))
            # Check down (r+2)
            if crow < self.H - 2:
                neighbors.append((crow + 2, ccol))
            # Check left (c-2)
            if ccol > 1:
                neighbors.append((crow, ccol - 2))
            # Check right (c+2)
            if ccol < self.W - 2:
                neighbors.append((crow, ccol + 2))

            if not neighbors:
                break  # Should not happen in this logic loop for valid grids

            # Pick a random neighbor
            idx = self.rng.integers(0, len(neighbors))
            nrow, ncol = neighbors[idx]

            if self.grid[nrow, ncol] == WALL:
                # Open wall between current and neighbor
                wall_r = (crow + nrow) // 2
                wall_c = (ccol + ncol) // 2
                self.grid[wall_r, wall_c] = EMPTY
                self.grid[nrow, ncol] = EMPTY
                num_visited += 1

            # Move to neighbor
            crow, ccol = nrow, ncol

    def step(self, state: tuple[int, int], action: int) -> tuple[int, int]:
        """
        Executes an action in the maze and returns the next state.

        If the action leads to a wall or outside the boundary, the agent stays in the current state.

        Args:
            state (tuple[int, int]): Current position (row, col).
            action (int): Action index (0: UP, 1: DOWN, 2: LEFT, 3: RIGHT).

        Returns:
            next_state (tuple[int, int]): The new position (row, col) after the action.
        """

        r, c = state
        dr, dc = 0, 0

        if action == UP:
            dr = -1
        elif action == DOWN:
            dr = 1
        elif action == LEFT:
            dc = -1
        elif action == RIGHT:
            dc = 1

        nr, nc = r + dr, c + dc

        # Check bounds (though walls usually surround the maze)
        if 0 <= nr < self.H and 0 <= nc < self.W:
            if self.grid[nr, nc] == EMPTY:
                return (nr, nc)

        return (r, c)

    def get_reward(self, state: tuple[int, int]) -> float:
        return self.rewards[state]


def value_iteration(
    maze: Maze, gamma: float = 0.9, theta: float = 1e-6
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Perform Value Iteration.

    Args:
        maze: The Maze environment.
        gamma: Discount factor.
        theta: Convergence threshold.

    Returns:
        V: Final value function, numpy array of shape (H, W).
        history: List of value functions at each iteration. (for visualization)
    """
    V = np.zeros((maze.H, maze.W))
    history = [V.copy()]

    while True:
        delta = 0.0
        new_V = V.copy()

        for r in range(maze.H):
            for c in range(maze.W):
                if maze.grid[r, c] == WALL:
                    continue

                # TODO: Implement Bellman Optimality Equation
                # Apply: V(s) <- max_a [ R(s) + gamma * V(s') ]
                # 1. Compute Q(s, a) for all actions.
                # 2. Update new_V[r, c] to be the maximum of Q values.
                # 3. Update delta to track the maximum change in value function.
                pass

        V = new_V  # pyright: ignore[reportConstantRedefinition]
        history.append(V.copy())
        if delta < theta:
            break

    return V, history


def policy_iteration(
    maze: Maze, gamma: float = 0.9, theta: float = 1e-6
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """
    Perform Policy Iteration.

    Args:
        maze: The Maze environment.
        gamma: Discount factor.
        theta: Convergence threshold for policy evaluation.

    Returns:
        policy: Optimal policy, numpy array of shape (H, W) containing action indices (0-3).
        V: Optimal value function, numpy array of shape (H, W).
        history: List of (policy, value function) tuples at each iteration. (for visualization)
    """
    # Initialize random policy
    policy = np.zeros((maze.H, maze.W), dtype=np.int32)
    for r in range(maze.H):
        for c in range(maze.W):
            if maze.grid[r, c] != WALL:
                policy[r, c] = np.random.choice(ACTIONS)

    V = np.zeros((maze.H, maze.W))
    history = [(policy.copy(), V.copy())]

    while True:
        # 1. Policy Evaluation
        while True:
            delta = 0.0
            new_V = V.copy()
            for r in range(maze.H):
                for c in range(maze.W):
                    if maze.grid[r, c] == WALL:
                        continue

                    # TODO: Implement Policy Evaluation (Bellman Expectation Equation)
                    # 1. Get action specified by current policy: a = policy[r, c]
                    # 2. Get next state (nr, nc) using `maze.step((r, c), a)`
                    # 3. Update V(s) = R(s) + gamma * V(s')
                    # 4. Update delta = max(delta, |V_new(s) - V(s)|)
                    pass

            V = new_V  # pyright: ignore[reportConstantRedefinition]
            if delta < theta:
                break

        # 2. Policy Improvement
        policy_stable = True
        for r in range(maze.H):
            for c in range(maze.W):
                if maze.grid[r, c] == WALL:
                    continue

                old_action = policy[r, c]

                # TODO: Implement Policy Improvement (Greedy Update)
                # 1. Compute Q-values for ALL actions: Q(s, a) = R(s) + gamma * V(s')
                # 2. Find the best action: argmax_a Q(s, a)
                # 3. If best_action != old_action:
                #    - Update policy[r, c]
                #    - Set policy_stable = False
                pass

        history.append((policy.copy(), V.copy()))

        if policy_stable:
            break

    return policy, V, history
