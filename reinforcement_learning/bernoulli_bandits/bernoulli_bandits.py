import random
from pathlib import Path
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.stats import beta


class BetaBernoulliArm(object):
    """
    Beta-Bernoulli arm tracking alpha / beta counts.

    Alpha is the prior success count, while beta is the prior failure count.

    Parameters
    ----------
    alpha : int
        Prior success count.
    beta : int
        Prior failure count.

    Attributes
    ----------
    alpha : int
        Success count plus prior.
    beta : int
        Failure count plus prior.
    """

    def __init__(self, alpha: int = 1, beta: int = 1) -> None:
        self.alpha: int = alpha
        self.beta: int = beta

    def sample(self) -> float:
        """
        Draw a probability sample from the Beta distribution.

        Returns
        -------
        float
            Sampled probability from Beta(alpha, beta).
        """
        return random.betavariate(self.alpha, self.beta)

    def update(self, reward: int) -> None:
        """
        Update counts based on observed reward.

        Parameters
        ----------
        reward : int
            Observed reward (1 for success, 0 for failure).
        """
        self.alpha += reward
        self.beta += 1 - reward


class EpsilonGreedyBernoulliBanditManager(object):
    """
    Manager for Bernoulli arms using epsilon-greedy selection.

    Parameters
    ----------
    actions : List[str]
        List of action identifiers.
    epsilon : float
        Probability of exploring (selecting a random arm).

    Attributes
    ----------
    arms : Dict[str, BetaBernoulliArm]
        Mapping from action name to its BetaBernoulliArm.
    epsilon : float
        Exploration probability.
    """

    def __init__(self, actions: List[str], epsilon: float = 0.1) -> None:
        self.arms: Dict[str, BetaBernoulliArm] = {
            action: BetaBernoulliArm() for action in actions
        }
        self.epsilon: float = epsilon

    def select_action(self) -> str:
        """
        Select an action using the epsilon-greedy strategy.

        With probability epsilon, choose a random action; otherwise, choose
        the action with highest estimated success probability (alpha / (alpha + beta)).

        Returns
        -------
        str
            The selected action identifier.
        """
        if random.random() < self.epsilon:
            return random.choice(list(self.arms.keys()))
        estimates: Dict[str, float] = {
            action: arm.alpha / (arm.alpha + arm.beta)
            for action, arm in self.arms.items()
        }
        return max(estimates, key=estimates.get)  # type: ignore[arg-type]

    def update(self, action: str, reward: int) -> None:
        """
        Update the chosen arm with the observed reward.

        Parameters
        ----------
        action : str
            The action identifier.
        reward : int
            Observed reward (1 for success, 0 for failure).
        """
        self.arms[action].update(reward)


class ThompsonSamplingBernoulliBanditManager(object):
    """
    Manager for Bernoulli arms using Thompson sampling.

    Parameters
    ----------
    actions : List[str]
        List of action identifiers.

    Attributes
    ----------
    arms : Dict[str, BetaBernoulliArm]
        Mapping from action name to its BetaBernoulliArm.
    """

    def __init__(self, actions: List[str]) -> None:
        self.arms: Dict[str, BetaBernoulliArm] = {
            action: BetaBernoulliArm() for action in actions
        }

    def select_action(self) -> str:
        """
        Select an action using Thompson sampling.

        Sample a theta from each arm's Beta posterior and pick the action
        with the highest sampled theta.

        Returns
        -------
        str
            The selected action identifier.
        """
        samples: Dict[str, float] = {
            action: arm.sample() for action, arm in self.arms.items()
        }
        return max(samples, key=samples.get)  # type: ignore[arg-type]

    def update(self, action: str, reward: int) -> None:
        """
        Update the chosen arm with the observed reward.

        Parameters
        ----------
        action : str
            The action identifier.
        reward : int
            Observed reward (1 for success, 0 for failure).
        """
        self.arms[action].update(reward)


def simulate_and_animate(
    manager: Union[
        EpsilonGreedyBernoulliBanditManager, ThompsonSamplingBernoulliBanditManager
    ],
    true_probs: Dict[str, float],
    n_rounds: int,
    interval: int,
    filename: Union[str, Path],
) -> None:
    """
    Simulate the bandit manager for a given number of rounds and animate the
    evolution of each arm's Beta distribution, saving to a GIF.

    Parameters
    ----------
    manager : Union[EpsilonGreedyBernoulliBanditManager, ThompsonSamplingBernoulliBanditManager]
        The bandit manager instance.
    true_probs : Dict[str, float]
        Ground-truth success probabilities for each action.
    n_rounds : int
        Total number of simulation rounds.
    interval : int
        Record state every `interval` rounds.
    filename : Union[str, Path]
        Output GIF filename or `Path` object.
    """
    actions: List[str] = list(true_probs.keys())
    params: Dict[str, Dict[str, List[int]]] = {
        a: {"alpha": [], "beta": []} for a in actions
    }

    # Record flat prior
    for a in actions:
        params[a]["alpha"].append(manager.arms[a].alpha)
        params[a]["beta"].append(manager.arms[a].beta)

    # Simulate n_rounds (i.e., time steps)
    for i in range(n_rounds):
        action: str = manager.select_action()
        reward: int = int(random.random() < true_probs[action])
        manager.update(action, reward)
        if (i + 1) % interval == 0:
            for a in actions:
                arm = manager.arms[a]
                params[a]["alpha"].append(arm.alpha)
                params[a]["beta"].append(arm.beta)

    frames: int = len(params[actions[0]]["alpha"])
    x: np.ndarray = np.linspace(0, 1, 200)

    fig, ax = plt.subplots()
    lines = {a: ax.plot([], [], label=a)[0] for a in actions}
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_ylabel("Density")
    ax.legend()

    def init() -> List[plt.Line2D]:
        for ln in lines.values():
            ln.set_data([], [])
        return list(lines.values())

    def update_frame(frame: int) -> List[plt.Line2D]:
        y_max: float = 0.0
        for a in actions:
            α, β = params[a]["alpha"][frame], params[a]["beta"][frame]
            y = beta(α, β).pdf(x)
            lines[a].set_data(x, y)
            y_max = max(y_max, float(y.max()))
        ax.set_ylim(0, y_max * 1.1)
        title = (
            "Prior (Beta(1, 1))" if frame == 0 else f"After Round {frame * interval}"
        )
        ax.set_title(title)
        return list(lines.values())

    ani = FuncAnimation(
        fig,
        update_frame,
        frames=frames,
        init_func=init,
        interval=500,
        blit=False,
        repeat=False,
    )
    ani.save(filename, writer="pillow", fps=2)
    plt.close(fig)
    print(f"Saved animation as '{filename}'")


def main() -> int:
    current_dir: Path = Path(__file__).parent
    actions: List[str] = ["A", "B", "C", "D"]
    true_probs: Dict[str, float] = {"A": 0.31, "B": 0.54, "C": 0.29, "D": 0.42}
    n_rounds, interval = 500, 20

    # Epsilon-greedy simulation
    eg_manager = EpsilonGreedyBernoulliBanditManager(actions, epsilon=0.1)
    simulate_and_animate(
        eg_manager, true_probs, n_rounds, interval, current_dir / "bandit_epsilon.gif"
    )

    # Thompson sampling simulation
    ts_manager = ThompsonSamplingBernoulliBanditManager(actions)
    simulate_and_animate(
        ts_manager, true_probs, n_rounds, interval, current_dir / "bandit_thompson.gif"
    )

    return 0


if __name__ == "__main__":
    main()
