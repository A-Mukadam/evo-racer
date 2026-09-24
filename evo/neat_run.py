"""NEAT neuroevolution with neat-python.

Same 8 inputs / 2 outputs as the GA; topology is evolved.
"""

import neat

TRAIN_TRACKS = ("oval", "rounded_rect", "wavy")  # TODO: share with ga_deap
CONFIG_PATH = "evo/neat_config.ini"               # TODO: create config file (num_inputs=8, num_outputs=2)


def make_config(path: str = CONFIG_PATH) -> neat.Config:
    """Load a neat-python config.

    Args:
        path: path to the NEAT .ini config.

    Returns:
        neat.Config with DefaultGenome, DefaultReproduction,
        DefaultSpeciesSet, DefaultStagnation.
    """
    raise NotImplementedError


def make_policy(genome, config: neat.Config):
    """Wrap a NEAT genome as a policy callable.

    Args:
        genome: neat.DefaultGenome.
        config: neat.Config.

    Returns:
        Callable obs (np.ndarray (8,)) -> (steer, throttle), each in [-1, 1].
    """
    raise NotImplementedError


def eval_genome(genome, config: neat.Config) -> float:
    """Fitness of one genome, for neat.ParallelEvaluator.

    Args:
        genome: neat.DefaultGenome.
        config: neat.Config.

    Returns:
        Aggregate fitness over TRAIN_TRACKS (same aggregation as ga_deap.evaluate).
    """
    # TODO: ParallelEvaluator doesn't pass a generation seed; need a way to
    # give all genomes in a generation the same episode seed.
    raise NotImplementedError


def run(config_path: str, seed: int, generations: int = 100) -> dict:
    """One full NEAT run.

    Args:
        config_path: path to NEAT config.
        seed: master RNG seed.
        generations: number of generations.

    Returns:
        {"best_genome": neat.DefaultGenome, "best_fitness": float,
         "stats": neat.StatisticsReporter, "evals": int}
    """
    raise NotImplementedError


def main() -> None:
    """CLI entry point: parse config path, seed, generations; run; save results/.

    Returns:
        None.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
