import pytest
import numpy as np
from rl import Maze, value_iteration, policy_iteration, EMPTY


@pytest.mark.public
def test_single_state_maze():
    """
    Test on a 1x1 maze (effective 3x3 grid with 1 empty cell).
    The agent is trapped in (1,1) with reward 10.
    V(s) = R(s) + gamma * V(s) -> V(s) = R(s) / (1 - gamma)
    """
    maze = Maze(1, 1, seed=42)
    gamma = 0.5
    
    # Sanity check environment
    assert maze.grid[1, 1] == EMPTY
    assert maze.rewards[1, 1] == 10.0
    
    V, _ = value_iteration(maze, gamma=gamma, theta=1e-6)
    expected_val = 10.0 / (1.0 - gamma) # 20.0
    
    assert np.isclose(V[1, 1], expected_val, rtol=1e-3)

@pytest.mark.public
def test_policy_iteration_consistency():
    """
    Test that policy iteration yields same value as value iteration.
    """
    maze = Maze(2, 2, seed=99)
    gamma = 0.9
    
    V_vi, _ = value_iteration(maze, gamma=gamma)
    _, V_pi, _ = policy_iteration(maze, gamma=gamma)
    
    # Ensure that the value function is not trivial (all zeros)
    # The goal reward is positive, so max value should be positive.
    assert np.max(V_vi) > 0.0, "Value function is all zeros; algorithm likely not implemented."

    # They should converge to the same optimal value function
    np.testing.assert_allclose(V_vi, V_pi, atol=1e-3)

@pytest.mark.public
def test_gamma_zero():
    """
    Test that when gamma is 0, V(s) equals the immediate reward R(s).
    V(s) = R(s) + 0 * ... = R(s)
    Check only for valid empty states.
    """
    maze = Maze(3, 3, seed=42)
    V, _ = value_iteration(maze, gamma=0.0)
    
    # Check only valid (empty) cells
    valid_mask = (maze.grid == EMPTY)
    
    # V should be exactly equal to rewards map at valid states
    np.testing.assert_allclose(V[valid_mask], maze.rewards[valid_mask], atol=1e-5)
