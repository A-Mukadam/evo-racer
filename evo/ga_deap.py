"""Fixed-topology neuroevolution with DEAP.

Network: 8 inputs -> 8 hidden (tanh) -> 2 outputs (steer, throttle).
Genome: flat list of 90 floats = W1 (8x8) + b1 (8) + W2 (8x2) + b2 (2).
Parallel evaluation: toolbox.register("map", pool.map).
"""

from dataclasses import dataclass

import numpy as np

N_INPUTS = 8
N_HIDDEN = 8
N_OUTPUTS = 2
GENOME_LEN = N_INPUTS * N_HIDDEN + N_HIDDEN + N_HIDDEN * N_OUTPUTS + N_OUTPUTS  # 90

TRAIN_TRACKS = ("oval", "rounded_rect", "wavy")
TEST_TRACKS = ("peanut",)


@dataclass
class GAConfig:
    """GA hyperparameters. Ablations vary pop_size, sigma, selection, elitism.

    Attributes:
        pop_size: individuals per generation.
        generations: number of generations.
        cx_prob: crossover probability per pair, [0, 1].
        mut_prob: mutation probability per individual, [0, 1].
        sigma: Gaussian mutation std dev, weight units.
        indpb: per-gene mutation probability, [0, 1].
        selection: "tournament" | "roulette" | ... (TODO: pick ablation set).
        tournsize: tournament size, if selection == "tournament".
        elitism: number of top individuals copied unchanged, >= 0.
        train_tracks: track names evaluated for fitness.
        processes: worker processes for multiprocessing.Pool.
    """

    pop_size: int = 100
    generations: int = 100
    cx_prob: float = 0.5
    mut_prob: float = 0.2
    sigma: float = 0.3
    indpb: float = 0.1
    selection: str = "tournament"
    tournsize: int = 3
    elitism: int = 1
    train_tracks: tuple[str, ...] = TRAIN_TRACKS
    processes: int | None = None


def decode(genome) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Unpack a flat genome into layer weights.

    Args:
        genome: sequence of GENOME_LEN floats.

    Returns:
        (W1 (8, 8), b1 (8,), W2 (8, 2), b2 (2,)) as float arrays.
    """
    raise NotImplementedError


def forward(params, obs: np.ndarray) -> tuple[float, float]:
    """Network forward pass.

    Args:
        params: output of decode().
        obs: np.ndarray shape (8,), float in [0, 1].

    Returns:
        (steer, throttle), each in [-1, 1] (tanh outputs).
    """
    raise NotImplementedError


def make_policy(genome):
    """Wrap a genome as a policy callable.

    Args:
        genome: sequence of GENOME_LEN floats.

    Returns:
        Callable obs (np.ndarray (8,)) -> (steer, throttle).
    """
    raise NotImplementedError


def evaluate(genome, track_names: tuple[str, ...], seed: int) -> tuple[float]:
    """DEAP fitness function. Must be picklable (top-level) for pool.map.

    Args:
        genome: sequence of GENOME_LEN floats.
        track_names: tracks to drive; one episode each.
        seed: generation's shared episode seed.

    Returns:
        1-tuple (fitness,): mean of per-track env.game fitness.
    """
    raise NotImplementedError


def build_toolbox(cfg: GAConfig, pool=None):
    """Register creator types and operators.

    Args:
        cfg: GAConfig.
        pool: multiprocessing.Pool or None (serial map).

    Returns:
        deap.base.Toolbox with attr, individual, population, evaluate, mate,
        mutate, select, and map registered.
    """
    # TODO: guard creator.create against re-registration across runs.
    raise NotImplementedError


def run(cfg: GAConfig, seed: int, pool=None) -> dict:
    """One full GA run.

    Args:
        cfg: GAConfig.
        seed: master RNG seed (random + numpy); also derives per-generation
            episode seeds.
        pool: multiprocessing.Pool or None.

    Returns:
        {"best_genome": list[float], "best_fitness": float,
         "logbook": deap.tools.Logbook (per-gen min/mean/max fitness),
         "evals": int  # total fitness evaluations, for matched-budget baselines}
    """
    raise NotImplementedError


def main() -> None:
    """CLI entry point: parse GAConfig fields + seed, run, save results/.

    Returns:
        None.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
