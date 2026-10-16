"""30-seed sweeps, baselines, held-out evaluation, and ablations.

Methods: "ga", "neat", "random" (random weights, matched eval budget),
"heuristic" (hand-coded controller). Results written under results/.
"""

import numpy as np

N_SEEDS = 30
RESULTS_DIR = "results"

# heuristic_policy tuning (searched on all 8 tracks x 10 seeds: 0 crashes)
STEER_GAIN = 3.0     # steer = gain * (free space on the right - free space on the left)
SLOW, FAST = 0.45, 1.0           # target speed range, as a fraction of MAX_SPEED
CLOSE, OPEN = 0.2, 0.7           # space ahead (normalized ray) at which target = SLOW / FAST


def heuristic_policy(obs: np.ndarray) -> tuple[float, float]:
    """Hand-coded baseline controller.

    Args:
        obs: np.ndarray shape (8,): 7 normalized sensors + normalized speed.

    Returns:
        (steer, throttle), each in [-1, 1].

    Steers toward the side with more free space. Picks a target speed from
    the shortest of the three forward rays (-20, 0, +20 deg) and brakes or
    accelerates toward it, so it slows for hairpins the GA champions don't.
    """
    sensors, speed = obs[:7], obs[7]
    steer = STEER_GAIN * ((sensors[4] + sensors[5] + sensors[6]) - (sensors[0] + sensors[1] + sensors[2]))
    ahead = min(sensors[2], sensors[3], sensors[4])
    target = SLOW + (FAST - SLOW) * np.clip((ahead - CLOSE) / (OPEN - CLOSE), 0.0, 1.0)
    return float(np.clip(steer, -1, 1)), float(np.clip(20.0 * (target - speed), -1, 1))


def random_baseline(budget: int, track_names: tuple[str, ...], seed: int) -> dict:
    """Best of `budget` uniformly random 90-weight genomes.

    Args:
        budget: number of fitness evaluations (match GA pop_size * generations).
        track_names: training tracks.
        seed: RNG seed.

    Returns:
        {"best_genome": list[float], "best_fitness": float,
         "curve": list[float]  # best-so-far after each evaluation}
    """
    raise NotImplementedError


def eval_holdout(policy, track_names: tuple[str, ...], seeds: list[int]) -> dict:
    """Evaluate a fixed policy on (possibly unseen) tracks.

    Args:
        policy: callable obs -> (steer, throttle).
        track_names: tracks to evaluate on.
        seeds: episode seeds; one episode per (track, seed).

    Returns:
        {track_name: {"mean": float, "std": float, "fitnesses": list[float]}}
    """
    raise NotImplementedError


def generalization_gap(train_scores: dict, test_scores: dict) -> float:
    """Train-vs-test performance gap.

    Args:
        train_scores: eval_holdout output on training tracks.
        test_scores: eval_holdout output on held-out tracks.

    Returns:
        Mean train fitness minus mean test fitness. Also report the gap in
        lap fractions (checkpoints passed / checkpoints per lap), since tracks
        have different checkpoint counts.
    """
    raise NotImplementedError


def sweep(method: str, cfg: dict, seeds: list[int]) -> list[dict]:
    """Run one method across many seeds.

    Args:
        method: "ga" | "neat" | "random" | "heuristic".
        cfg: method-specific config overrides.
        seeds: master seeds, one run each.

    Returns:
        One result dict per seed (best fitness, convergence curve, holdout scores).
    """
    raise NotImplementedError


def ablation_grid() -> list[tuple[str, dict]]:
    """GA ablation configurations.

    Returns:
        List of (label, cfg overrides) varying pop_size, sigma, selection,
        elitism one factor at a time from the default GAConfig.
    """
    # TODO: pick levels for each factor.
    raise NotImplementedError


def save_results(name: str, results: list[dict]) -> str:
    """Persist sweep results.

    Args:
        name: run label, used in the filename.
        results: sweep() output.

    Returns:
        Path of the written file under RESULTS_DIR.
    """
    raise NotImplementedError


def plot_convergence(results_by_method: dict[str, list[dict]], out_path: str) -> None:
    """Mean +/- std best-fitness curves per method.

    Args:
        results_by_method: {method: sweep() output}.
        out_path: PNG path.

    Returns:
        None.
    """
    raise NotImplementedError


def main() -> None:
    """CLI entry point: choose method(s), seeds, ablations; run and plot.

    Returns:
        None.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
