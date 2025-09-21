from typing import Dict, List, Optional, Tuple, TypeAlias

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

rng: np.random.Generator = np.random.default_rng(seed=1227)

BetaPrior: TypeAlias = Tuple[float, float]
MeanReward: TypeAlias = float
ArmLabel: TypeAlias = str


class DynamicBanditConfig(BaseModel):
    """
    Configuration for the dynamic bandit algorithm.

    Attributes
    ----------
    initial_means : List[MeanReward]
        The initial mean rewards for each arm.
    initial_priors : List[BetaPrior]
        The initial (alpha, beta) priors for each arm in the same order as `initial_means`.
    batch_new_arms : Dict[int, Dict[ArmLabel, MeanReward]]
        A mapping of time step to new arms introduced at that time, with arm labels and their means.
    batch_new_priors : Dict[int, Dict[ArmLabel, BetaPrior]]
        Mapping of time step to new arms' (alpha, beta) priors introduced at that time.
        Any missing `arm_label` at a given `t` uses Beta(1, 1) as the prior.
    horizon : int
        The total number of time steps.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    initial_means: List[MeanReward] = Field(
        ..., description="Initial mean rewards for each arm."
    )
    initial_priors: List[BetaPrior] = Field(
        ..., description="Initial Beta priors (alpha, beta) for each arm."
    )
    batch_new_arms: Dict[int, Dict[ArmLabel, MeanReward]] = Field(
        ..., description="Time step -> new arms with mean rewards."
    )
    batch_new_priors: Dict[int, Dict[ArmLabel, BetaPrior]] = Field(
        ..., description="Time step -> new arms with Beta priors."
    )
    horizon: int = Field(
        ..., ge=1, description="Total number of time steps (must be >= 1)."
    )


class DynamicBernoulliBandit(object):
    """
    Dynamic Bernoulli bandit environment supporting batch arm arrivals.

    This class models a non-stationary multi-armed bandit where new arms can
    be introduced at specific time steps according to a user-supplied schedule.
    Each arm returns Bernoulli rewards with fixed but unknown mean.

    Attributes
    ----------
    rng : numpy.random.Generator
        Pseudo-random number generator used for reward draws.
    horizon : int
        Total number of time steps for the bandit horizon.
    _batch : Dict[int, Dict[ArmLabel, MeanReward]]
        Mapping from time t -> {arm_label: true_mean} defining when new arms are added.
    arm_means : List[MeanReward]
        List of true success probabilities for currently available arms.
    arm_labels : List[ArmLabel]
        Human-readable labels for each arm (initial arms labeled "init_0", "init_1", ...).
    """

    def __init__(self, config: DynamicBanditConfig, rng: np.random.Generator) -> None:
        """
        Initialize the dynamic Bernoulli bandit with the given configuration and random number generator.

        Parameters
        ----------
        config : DynamicBanditConfig
            Configuration specifying initial arm means, arm arrival schedule, their respective priors, and time horizon.
        rng : numpy.random.Generator
            Pseudo-random number generator for sampling rewards.

        Returns
        -------
        None
        """
        self.rng: np.random.Generator = rng
        self.horizon: int = int(config.horizon)
        self._batch: Dict[int, Dict[str, float]] = config.batch_new_arms
        self.arm_means: List[float] = list(config.initial_means)
        self.arm_labels: List[str] = [f"init_{i}" for i in range(len(self.arm_means))]

    def add_arms(self, t: int) -> List[Optional[int]]:
        """
        Add any arms scheduled to appear at time t.

        Parameters
        ----------
        t : int
            Current time step.

        Returns
        -------
        List[Optional[int]]
            Indices of the arms that were added at this step. If no new arms are scheduled for time t, returns an empty list.
        """
        new_indices: List[Optional[int]] = []
        # Check if any arms are scheduled to be added at time t
        if t in self._batch:
            for label, mean in self._batch[t].items():
                self.arm_means.append(float(mean))
                self.arm_labels.append(str(label))
                new_indices.append(len(self.arm_means) - 1)
        return new_indices

    def pull(self, arm: int) -> int:
        """
        Pull an arm and sample a Bernoulli outcome. This is effectively simulating a Bernoulli(p) random variable
        where `p` is the true mean of the specified arm.

        Parameters
        ----------
        arm : int
            Index of the arm to pull (0-based).

        Returns
        -------
        int
            Reward outcome from Bernoulli(p), where p is the true mean of the arm.
        """
        p: float = self.arm_means[arm]
        return int(self.rng.random() < p)

    def optimal_mean(self) -> float:
        """
        Compute the true mean of the best currently available arm. This
        is used to compute instantaneous regret.

        Returns
        -------
        float
            Maximum of all arm success probabilities in `arm_means`.
        """
        return float(max(self.arm_means))


class ThompsonSampling(object):
    """
    Thompson Sampling policy for Bernoulli multi-armed bandits with dynamic arms.

    This implementation maintains independent Beta posteriors for each arm and
    uses random sampling from these posteriors to select arms. It supports
    dynamically adding new arms during execution and initializes them with
    a specified Beta prior.

    Attributes
    ----------
    alpha : List[float]
        Alpha parameters for the Beta posterior of each arm.
    beta : List[float]
        Beta parameters for the Beta posterior of each arm.
    """

    def __init__(
        self,
        init_priors: List[BetaPrior],
    ) -> None:
        """
        Parameters
        ----------
        init_priors : List[BetaPrior]

        Returns
        -------
        None
        """
        self.alpha: List[float] = [float(a) for a, _ in init_priors]
        self.beta: List[float] = [float(b) for _, b in init_priors]

    def add_new_arms(self, priors: List[BetaPrior]) -> None:
        """
        Append new arms with given priors.

        Parameters
        ----------
        priors : List[BetaPrior]
            Priors (alpha, beta) for new arms to be added.

        Returns
        -------
        None
        """
        for a, b in priors:
            self.alpha.append(float(a))
            self.beta.append(float(b))

    def select_arm(self, rng: np.random.Generator) -> int:
        """
        Select an arm using Thompson Sampling.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator used to draw samples from the Beta posteriors.

        Returns
        -------
        int
            Index of the arm selected (0-based).
        """
        samples: List[float] = [rng.beta(a, b) for a, b in zip(self.alpha, self.beta)]
        return int(np.argmax(samples))

    def update(self, arm_index: int, outcome: int) -> None:
        """
        Update the posterior parameters after observing a reward.

        Parameters
        ----------
        arm : int
            Index of the arm that was pulled.
        outcome : int
            Observed outcome (1 for success, 0 for failure).

        Returns
        -------
        None
        """
        self.alpha[arm_index] += outcome
        self.beta[arm_index] += 1 - outcome


class SimulationResult(BaseModel):
    """
    Container for the results of a single Thompson Sampling simulation run.

    Attributes
    ----------
    outcomes : npt.NDArray[np.floating]
        Realized Bernoulli outcomes at each time step.
    instantaneous_regrets : npt.NDArray[np.floating]
        Instantaneous regret at each time step.
    cumulative_regrets : npt.NDArray[np.floating]
        Cumulative sum of instantaneous regret over time.
    chosen_arm_indices : npt.NDArray[np.int_]
        Indices of arms selected at each time step.
    num_arms_hist : npt.NDArray[np.int_]
        Number of available arms at each time step.
    arm_labels : List[str]
        Labels for all arms in index order at the end of the run.
    arrivals_map : Dict[int, List[Optional[int]]]
        Mapping from arrival time t to indices of newly added arms.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    outcomes: npt.NDArray[np.floating] = Field(...)
    instantaneous_regrets: npt.NDArray[np.floating] = Field(...)
    cumulative_regrets: npt.NDArray[np.floating] = Field(...)
    chosen_arm_indices: npt.NDArray[np.int_] = Field(...)
    num_arms_hist: npt.NDArray[np.int_] = Field(...)
    arm_labels: List[str] = Field(...)
    arrivals_map: Dict[int, List[Optional[int]]] = Field(...)


def _priors_for_added_labels(
    t: int,
    labels_added: List[ArmLabel],
    config: DynamicBanditConfig,
) -> List[BetaPrior]:
    """
    Resolve per-arm priors for labels added at time t.

    Parameters
    ----------
    t : int
        Current time step.
    labels_added : List[ArmLabel]
        Labels of arms added at time t.
    config : DynamicBanditConfig
        Configuration for the dynamic bandit.

    Raises
    ------
    KeyError
        If a required label prior is missing at time t.

    Returns
    -------
    List[BetaPrior]
        Priors aligned with `labels_added`.
    """
    priors_map_t = config.batch_new_priors.get(t, {})
    priors: List[BetaPrior] = []
    for label in labels_added:
        if label not in priors_map_t:
            raise KeyError(f"Missing Beta prior for new arm '{label}' at t = {t} ")
        a, b = priors_map_t[label]
        priors.append((float(a), float(b)))
    return priors


def simulate_once(
    config: DynamicBanditConfig, rng: np.random.Generator
) -> SimulationResult:
    """
    Run a single Thompson Sampling trajectory in a dynamic Bernoulli bandit with batch arrivals.

    Parameters
    ----------
    config : DynamicBanditConfig
        Environment and policy configuration. Must include initial arm means, arm arrival schedule, their respective priors, and time horizon.
    rng : numpy.random.Generator
        Pseudorandom number generator for sampling rewards.

    Returns
    -------
    SimulationResult

    Notes
    -----
    - The environment can add multiple new arms at a single time step.
    - New arms are initialized in the policy with specified Beta(alpha, beta) priors.
    - Instantaneous regret uses the true means to define the benchmark best arm at each time. This is standard for simulation analysis.
    """
    env: DynamicBernoulliBandit = DynamicBernoulliBandit(config, rng)
    ts_policy: ThompsonSampling = ThompsonSampling(init_priors=config.initial_priors)

    horizon: int = config.horizon
    outcomes: npt.NDArray[np.floating] = np.zeros(horizon, dtype=float)
    instantaneous_regrets: npt.NDArray[np.floating] = np.zeros(horizon, dtype=float)
    cumulative_regrets: npt.NDArray[np.floating] = np.zeros(horizon, dtype=float)
    chosen_arm_indices: npt.NDArray[np.int_] = np.zeros(horizon, dtype=int)
    num_arms_hist: npt.NDArray[np.int_] = np.zeros(horizon, dtype=int)

    # Track which arm indices arrive at each time for this run
    arrivals_map: Dict[int, List[Optional[int]]] = {}

    for time_step in range(horizon):
        # Apply scheduled arrivals and record indices of added arms
        added_indices: List[Optional[int]] = env.add_arms(time_step)
        if added_indices:
            arrivals_map[time_step] = added_indices
            # Resolve priors for these labels in the same order they were appended
            added_labels: List[ArmLabel] = [
                env.arm_labels[j] for j in added_indices if j is not None
            ]
            added_priors: List[BetaPrior] = _priors_for_added_labels(
                time_step, added_labels, config
            )
            # Extend TS posteriors so the new arms can be sampled and selected
            ts_policy.add_new_arms(added_priors)

        # Thompson step: sample from posteriors, pick the best sample, then pull
        arm_index: int = ts_policy.select_arm(rng)
        outcome_t: int = env.pull(arm_index)

        # Bayesian update of the chosen arm's Beta posterior with the new outcome (0 or 1)
        ts_policy.update(arm_index=arm_index, outcome=outcome_t)

        outcomes[time_step] = outcome_t
        best_mean_t: MeanReward = env.optimal_mean()
        chosen_mean_t: MeanReward = env.arm_means[arm_index]
        # Regret uses the gap between the best true mean at `t` and the chosen arm's true mean
        instantaneous_regrets[time_step] = best_mean_t - chosen_mean_t
        cumulative_regrets[time_step] = instantaneous_regrets[: time_step + 1].sum()
        chosen_arm_indices[time_step] = arm_index
        num_arms_hist[time_step] = len(env.arm_means)

    return SimulationResult(
        outcomes=outcomes,
        instantaneous_regrets=instantaneous_regrets,
        cumulative_regrets=cumulative_regrets,
        chosen_arm_indices=chosen_arm_indices,
        num_arms_hist=num_arms_hist,
        arm_labels=env.arm_labels,
        arrivals_map=arrivals_map,
    )


class AggregationResult(BaseModel):
    """
    Aggregated results across Monte Carlo runs.

    Attributes
    ----------
    mean_cum_regrets : npt.NDArray[np.floating]
        Average cumulative regret over time (shape: (T,)).
    new_arm_meta : List[Tuple[int, int, str]]
        Metadata tuples (arrival_time, arm_index, label) for every newly added arm.
    adoption_averages : Dict[Tuple[int, int], npt.NDArray[np.floating]]
        For each new arm keyed by (arrival_time, arm_index), an array of length `window_size`
        with the average cumulative selection rate within the first k steps after arrival.
        For arms arriving late, values beyond the remaining horizon are padded by the last
        valid cumulative rate.
    discovery_lags_averages : Dict[Tuple[int, int], float]
        For each new arm keyed by (arrival_time, arm_index), the average discovery lag in steps.
    window_size : int
        The window length used for adoption statistics.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    mean_cum_regrets: npt.NDArray[np.floating] = Field(...)
    new_arm_meta: List[Tuple[int, int, str]] = Field(...)
    adoption_averages: Dict[Tuple[int, int], npt.NDArray[np.floating]] = Field(...)
    discovery_lags_averages: Dict[Tuple[int, int], float] = Field(...)
    window_size: int = Field(...)


def simulate_many(
    n_runs: int,
    template_rng: np.random.Generator,
    config: DynamicBanditConfig,
    window_size: int = 60,
) -> AggregationResult:
    """
    Average multiple Thompson Sampling trajectories in a dynamic Bernoulli bandit with batch arrivals.

    Runs `simulate_once` repeatedly, aggregates time series across runs, and computes per-new-arm
    adoption statistics and discovery lags. Adoption windows are capped by the remaining horizon to
    avoid overcounting late arrivals. The adoption curves are then padded to `window_size` by
    repeating the last valid cumulative rate to preserve a uniform shape across arms.

    Parameters
    ----------
    n_runs : int
        Number of independent Monte Carlo runs to average over.
    template_rng : numpy.random.Generator
        RNG used for the template run to fix (arrival_time -> arm indices) mapping.
    config : DynamicBanditConfig
        Environment and policy configuration.
    window_size : int, default=60
        Post-arrival window length used to compute cumulative selection rates.

    Returns
    -------
    AggregationResult
        Aggregated metrics and metadata across runs.
    """
    T: int = int(config.horizon)

    # Matrices: rows = runs, cols = time
    cumulative_regrets_matrix: npt.NDArray[np.floating] = np.zeros(
        (n_runs, T), dtype=float
    )
    chosen_arm_indices_matrix: npt.NDArray[np.int_] = np.zeros((n_runs, T), dtype=int)

    # Template run to fix mapping (arrival_time -> arm indices) used in aggregation
    template_result: SimulationResult = simulate_once(config=config, rng=template_rng)
    arrivals_map: Dict[int, List[Optional[int]]] = template_result.arrivals_map
    arm_labels: List[ArmLabel] = template_result.arm_labels

    # Metadata for new arms: (arrival_time, arm_index, arm_label)
    new_arm_meta: List[Tuple[int, int, ArmLabel]] = [
        (arrival_time, arm_index, arm_labels[arm_index])
        for arrival_time, arm_indices in arrivals_map.items()
        for arm_index in arm_indices
        if arm_index is not None
    ]

    # Per-new-arm collectors across runs
    adoption_series: Dict[Tuple[int, int], List[npt.NDArray[np.floating]]] = {
        (arrival_time, arm_index): [] for (arrival_time, arm_index, _) in new_arm_meta
    }
    discovery_lags: Dict[Tuple[int, int], List[int]] = {
        (arrival_time, arm_index): [] for (arrival_time, arm_index, _) in new_arm_meta
    }

    # Monte Carlo loop
    for run_id in range(n_runs):
        rng_i = np.random.default_rng(seed=run_id)
        res_i: SimulationResult = simulate_once(config=config, rng=rng_i)

        cumulative_regrets_matrix[run_id] = res_i.cumulative_regrets
        chosen_arm_indices_matrix[run_id] = res_i.chosen_arm_indices

        # Compute lag and adoption for each template new arm
        for arrival_time, arm_index, _ in new_arm_meta:
            # These are the arm choices from `arrival_time` until the end of the horizon
            post_choices: npt.NDArray[np.int_] = chosen_arm_indices_matrix[
                run_id, arrival_time:
            ]

            # Discovery lag: first offset where arm_index is chosen; max lag if never chosen
            hits: np.ndarray = np.where(post_choices == arm_index)[0]
            lag_steps: int = int(hits[0]) if hits.size > 0 else (T - arrival_time)
            discovery_lags[(arrival_time, arm_index)].append(lag_steps)

            # Adoption curve: cap by remaining horizon, then pad to `window_size` if needed
            remaining: int = T - arrival_time
            window_eff: int = int(min(window_size, remaining))
            if window_eff == 0:
                # Arrived at the final step: adoption curve is all zeros
                padded: npt.NDArray[np.floating] = np.zeros(window_size, dtype=float)
                adoption_series[(arrival_time, arm_index)].append(padded)
                continue

            # These are the first `window_eff` choices after `arrival_time`
            window: npt.NDArray[np.int_] = post_choices[:window_eff]
            cumulative_rate: npt.NDArray[np.floating] = np.zeros(
                window_eff, dtype=float
            )
            for k in range(window_eff):
                # Fraction of selections in the first k + 1 post-arrival trials
                cumulative_rate[k] = float(np.mean(window[: k + 1] == arm_index))

            # Pad to window_size by repeating the last valid rate
            if window_eff < window_size:
                pad_tail: npt.NDArray[np.floating] = np.full(
                    window_size - window_eff, cumulative_rate[-1], dtype=float
                )
                cumulative_rate = np.concatenate([cumulative_rate, pad_tail], axis=0)

            adoption_series[(arrival_time, arm_index)].append(cumulative_rate)

    # Aggregate across runs
    mean_cum_regret: npt.NDArray[np.floating] = cumulative_regrets_matrix.mean(axis=0)

    adoption_averages: Dict[Tuple[int, int], npt.NDArray[np.floating]] = {
        key: np.mean(np.stack(series, axis=0), axis=0)
        for key, series in adoption_series.items()
    }
    discovery_lags_averages: Dict[Tuple[int, int], float] = {
        key: float(np.mean(lags)) for key, lags in discovery_lags.items()
    }

    return AggregationResult(
        mean_cum_regrets=mean_cum_regret,
        new_arm_meta=[(t0, j, str(lbl)) for (t0, j, lbl) in new_arm_meta],
        adoption_averages=adoption_averages,
        discovery_lags_averages=discovery_lags_averages,
        window_size=int(window_size),
    )
