from typing import List, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


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


def main(
    print_tables: bool = True,
    show_plots: bool = True,
    actions: List[int] = None,
    rewards: List[int] = None,
) -> Union[int, Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]]:
    if actions is None and rewards is None:
        # Define the forced sequence of actions (k = 4) and rewards
        actions = [1, 2, 2, 2, 3, 4, 4, 3, 1, 2, 3, 3]
        rewards = [-1, 1, -2, 2, 0, 1, -1, 2, 0, 1, -2, 2]
        num_steps = len(actions)
        num_actions = 4
    else:
        if actions is None or rewards is None:
            raise ValueError("Both actions and rewards must be provided together")
        if len(actions) != len(rewards):
            raise ValueError("Actions and rewards lists must have the same length")
        if min(actions) < 1:
            raise ValueError("Actions must be 1-based integers, 1, ..., k")
        num_steps = len(actions)
        num_actions = max(actions)

    # Q[a - 1] will hold the current value estimate for action 'a'
    q_values = np.zeros(num_actions, dtype=float)
    action_counts = np.zeros(num_actions, dtype=int)

    q_value_history = []  # Q-values after each step t
    chosen_actions_history = []
    rewards_history = []

    # Store data for step-by-step analysis table
    step_analysis = []

    # Step through each (action, reward) pair
    for t in range(num_steps):
        action = actions[t]
        reward = rewards[t]

        # Identify which actions are currently greedy
        greedy_set_indices = find_greedy_actions(q_values)

        # Check if action is in the greedy set (action is 1-based, q_values index is 0-based)
        action_index = action - 1

        # Decide whether this step 'must' or 'possibly' be epsilon
        if action_index not in greedy_set_indices:
            # If the chosen action is not in the greedy set, it must be an epsilon action
            must_be_epsilon = "Yes"
            possibly_epsilon = "No"
        else:
            # If the chosen action is in the greedy set, it could be greedy or epsilon
            must_be_epsilon = "No"
            possibly_epsilon = "Yes"

        # Store data for the step analysis table
        step_analysis.append(
            {
                "Step": t + 1,
                "Chosen Action": action,
                "Reward": reward,
                "Greedy Set Before Step": [idx + 1 for idx in greedy_set_indices],
                "Must be Epsilon?": must_be_epsilon,
                "Possibly Epsilon?": possibly_epsilon,
            }
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

    step_analysis_data = pl.DataFrame(step_analysis)

    q_value_summaries = []
    for step_i, (act, rew, qvals) in enumerate(
        zip(chosen_actions_history, rewards_history, q_value_history), start=1
    ):
        q_value_summaries.append(
            {
                "Step": step_i,
                "Chosen": act,
                "Reward": rew,
                "Q1": round(qvals[0], 2),
                "Q2": round(qvals[1], 2),
                "Q3": round(qvals[2], 2),
                "Q4": round(qvals[3], 2),
            }
        )

    q_value_data = pl.DataFrame(q_value_summaries)

    final_q_values = q_value_history[-1]
    action_selection_counts = [
        chosen_actions_history.count(i + 1) for i in range(num_actions)
    ]

    summaries = []
    for i in range(num_actions):
        summaries.append(
            {
                "Action": i + 1,
                "Final Q-Value": round(final_q_values[i], 3),
                "Times Selected": action_selection_counts[i],
                "Average Reward": round(final_q_values[i], 3)
                if action_selection_counts[i] > 0
                else 0,
            }
        )

    summary_data = pl.DataFrame(summaries)

    if print_tables:
        print("Step-by-Step Analysis:")
        with pl.Config(tbl_rows=len(step_analysis_data)):
            print(step_analysis_data, "\n")

        print("Detailed Q-value Table (after each update):")
        with pl.Config(tbl_rows=len(q_value_data)):
            print(q_value_data, "\n")

        print("Summary Statistics:")
        with pl.Config(tbl_rows=len(summary_data)):
            print(summary_data, "\n")

    if show_plots:
        # (A) Plot Q-values over time
        time_axis = range(1, num_steps + 1)
        plt.figure()  # separate figure
        for a_idx in range(num_actions):
            # Extract the Q-values of action 'a_idx' over time
            q_values_for_this_action = [
                q_history[a_idx] for q_history in q_value_history
            ]
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

    if not print_tables:
        return step_analysis_data, q_value_data, summary_data
    return 0


if __name__ == "__main__":
    main()
