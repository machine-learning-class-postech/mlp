# type: ignore
from typing import Any, Union

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from rl import DOWN, LEFT, RIGHT, UP, WALL, Maze, policy_iteration, value_iteration


def get_vi_history(
    maze: Maze, gamma: float = 0.9, theta: float = 1e-6
) -> list[np.ndarray]:
    """
    Returns a history with just the initial and final Value functions.
    Uses the student's implementation of value_iteration.
    """

    # Run student's implementation to get final V
    _, history = value_iteration(maze, gamma, theta)

    # Return history: Start -> End
    # We can repeat frames to make it clearer if needed, but the animation loop handles delays.
    return history


def get_pi_history(
    maze: Maze, gamma: float = 0.9, theta: float = 1e-6
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Returns a history with just the initial and final (Policy, V).
    Uses the student's implementation of policy_iteration.
    """

    # Run student's implementation to get final Policy and V
    _, _, history = policy_iteration(maze, gamma, theta)

    # Return history: Start -> End
    return history


def create_animation(
    maze: Maze,
    history: Union[list[np.ndarray], list[tuple[np.ndarray, np.ndarray]]],
    filename: str,
    title_prefix: str,
    is_policy: bool = False,
    end_hold_frames: int = 5,
):
    """
    Generates and saves a GIF animation from the history.
    """
    H, W = maze.H, maze.W
    fig, ax = plt.subplots(figsize=(6, 6))

    # Setup static maze components (walls, goal)
    # We will redraw these or keep them persistent

    # Calculate global min/max for consistent color scaling (only used for Value Iteration)
    all_vs: list[np.ndarray] = []
    if not is_policy:
        for item in history:
            all_vs.append(item)

        if all_vs:
            min_val = np.min([np.min(v) for v in all_vs])
            max_val = np.max([np.max(v) for v in all_vs])
        else:
            min_val, max_val = None, None

    def plot_frame(frame_data: dict[str, Any]):
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{title_prefix} - Step {frame_data['step']}")

        V = frame_data["V"]
        policy = frame_data.get("policy")

        if not is_policy:
            # Mask walls for heatmap
            masked_V = np.ma.masked_where(maze.grid == WALL, V)
            ax.imshow(
                masked_V, cmap="viridis", origin="upper", vmin=min_val, vmax=max_val
            )
        else:
            # No heatmap for Policy Iteration, just set limits
            ax.set_xlim(-0.5, W - 0.5)
            ax.set_ylim(H - 0.5, -0.5)

        # Draw Walls and Goal
        for r in range(H):
            for c in range(W):
                if maze.grid[r, c] == WALL:
                    rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="black")
                    ax.add_patch(rect)
                if maze.rewards[r, c] > 0:
                    rect = plt.Rectangle(
                        (c - 0.5, r - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="lime",
                        linewidth=3,
                    )
                    ax.add_patch(rect)
                    ax.text(
                        c,
                        r,
                        "G",
                        ha="center",
                        va="center",
                        color="lime",
                        fontweight="bold",
                    )

        # Draw Arrows (for Policy Iteration or final VI if desired, but VI history usually just V)
        if is_policy and policy is not None:
            for r in range(H):
                for c in range(W):
                    if maze.grid[r, c] == WALL:
                        continue

                    # Skip Goal state for arrows
                    if maze.rewards[r, c] > 0:
                        continue

                    action = policy[r, c]
                    dx, dy = 0, 0
                    if action == UP:
                        dy = -0.3
                    elif action == DOWN:
                        dy = 0.3
                    elif action == LEFT:
                        dx = -0.3
                    elif action == RIGHT:
                        dx = 0.3

                    if dx != 0 or dy != 0:
                        ax.arrow(
                            c,
                            r,
                            dx,
                            dy,
                            head_width=0.2,
                            head_length=0.2,
                            fc="black",
                            ec="black",
                        )

    frames = []
    for i, item in enumerate(history):
        data = {"step": i}
        if is_policy:
            data["policy"] = item[0]
            data["V"] = item[1]
        else:
            data["V"] = item
        frames.append(data)

    # Add extra frames for holding the last state
    if frames:
        last_frame_data = frames[-1].copy()
        for _ in range(end_hold_frames):
            frames.append(last_frame_data)

    ani = animation.FuncAnimation(
        fig, plot_frame, frames=frames, interval=200, repeat_delay=2000
    )

    # Save as GIF using PillowWriter
    writer = animation.PillowWriter(fps=5)
    ani.save(filename, writer=writer)
    print(f"Saved animation to {filename}")

    # Save colorbar for Value Iteration
    if not is_policy and min_val is not None:
        fig_scale, ax_scale = plt.subplots(figsize=(6, 1))
        fig_scale.subplots_adjust(bottom=0.5)
        norm = plt.Normalize(vmin=min_val, vmax=max_val)
        cb = fig_scale.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap="viridis"),
            cax=ax_scale,
            orientation="horizontal",
            label="Value",
        )
        fig_scale.savefig("value_scale.png")
        plt.close(fig_scale)
        print("Saved value scale to value_scale.png")

    plt.close(fig)


if __name__ == "__main__":
    # Generate Maze
    maze = Maze(5, 5, seed=42)

    print("Generating Value Iteration Animation...")
    # Use a larger theta for visualization to reduce animation length (visual convergence)
    vi_history = get_vi_history(maze, theta=1e-3)
    create_animation(
        maze, vi_history, "value_iteration.gif", "Value Iteration", is_policy=False
    )

    print("Generating Policy Iteration Animation...")
    pi_history = get_pi_history(maze, theta=1e-3)
    create_animation(
        maze, pi_history, "policy_iteration.gif", "Policy Iteration", is_policy=True
    )
