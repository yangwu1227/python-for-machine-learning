from typing import Set

import matplotlib.pyplot as plt
import numpy as np


def find_greedy_actions(current_q: np.ndarray) -> Set[int]:
    """
    Function to find the set of all greedy actions given current Q-values at
    a particular time step.

    Parameters
    ----------
    current_q : np.ndarray
        A numpy array of current Q-values for each action.

    Returns
    -------
    Set[int]
        A set of indices (0-based) for all actions with the maximal Q-value.
    """
    max_q = np.max(current_q)
    return {i for i, v in enumerate(current_q) if v == max_q}


def main() -> int:
    # Define the forced sequence of actions (k = 4) and rewards
    actions = [1, 2, 2, 2, 3]
    rewards = [-1, 1, -2, 2, 0]
    num_steps = len(actions)
    num_actions = 4

    # Q[a - 1] will hold the current value estimate for action 'a'
    q_values = np.zeros(num_actions, dtype=float)
    action_counts = np.zeros(num_actions, dtype=int)

    q_value_history = []  # Q-values after each step t
    chosen_actions_history = []
    rewards_history = []

    # Step through each (action, reward) pair
    print(
        f"{'Step':<5}{'Chosen Action':<15}{'Reward':<8}"
        f"{'Greedy set before step':<25}{'Must be Epsilon?':<18}{'Possibly Epsilon?'}"
    )

    for t in range(num_steps):
        action = actions[t]
        reward = rewards[t]

        # Identify which actions are currently greedy
        greedy_set_indices = find_greedy_actions(q_values)

        # Check if action is in the greedy set (action is 1-based, q_values index is 0-based)
        action_index = action - 1

        # Decide whether it 'must' or 'possibly' be epsilon
        if action_index not in greedy_set_indices:
            must_be_epsilon = "Yes"
            possibly_epsilon = "No"
        else:
            must_be_epsilon = "No"
            possibly_epsilon = "Yes"

        # Print a line describing this time step's forced action
        print(
            f"{t + 1:<5}{action:<15}{reward:<8}"
            f"{[idx + 1 for idx in greedy_set_indices]!s:<25}{must_be_epsilon:<18}{possibly_epsilon}"
        )

        # Update the Q-value estimate for the chosen action
        action_counts[action_index] += 1
        q_values[action_index] += (reward - q_values[action_index]) / action_counts[
            action_index
        ]

        # Record post-update data
        q_value_history.append(q_values.copy())
        chosen_actions_history.append(action)
        rewards_history.append(reward)

    print("\nDetailed Q-value table (after each update):")
    print(f"{'Step':<5}{'Chosen':<7}{'Reward':<7}{'Q1':>8}{'Q2':>8}{'Q3':>8}{'Q4':>8}")
    for step_i, (act, rew, qvals) in enumerate(
        zip(chosen_actions_history, rewards_history, q_value_history), start=1
    ):
        print(
            f"{step_i:<5}{act:<7}{rew:<7}"
            f"{qvals[0]:>8.2f}{qvals[1]:>8.2f}{qvals[2]:>8.2f}{qvals[3]:>8.2f}"
        )

    # (A) Plot Q-values over time
    time_axis = range(1, num_steps + 1)
    plt.figure()  # separate figure
    for a_idx in range(num_actions):
        # Extract the Q-values of action 'a_idx' over time
        q_values_for_this_action = [q_history[a_idx] for q_history in q_value_history]
        plt.plot(time_axis, q_values_for_this_action, label=f"Action {a_idx + 1}")
    plt.xlabel("Step")
    plt.ylabel("Estimated Q-Value")
    plt.xticks(time_axis)
    plt.title("Q-Value Estimates Over Time")
    plt.legend()
    plt.show()

    # (B) Plot reward at each step
    plt.figure()
    plt.plot(time_axis, rewards_history, marker="o")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.xticks(time_axis)
    plt.title("Reward Received at Each Step")
    plt.show()

    # (C) Plot chosen action at each step
    plt.figure()
    plt.plot(time_axis, chosen_actions_history, marker="x")
    plt.xlabel("Step")
    plt.ylabel("Chosen Action")
    plt.xticks(time_axis)
    plt.title("Actions Taken Over Time")
    plt.show()

    return 0


if __name__ == "__main__":
    main()
