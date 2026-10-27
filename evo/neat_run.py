"""NEAT neuroevolution with neat-python.

Same 8 inputs / 2 outputs as the GA; topology is evolved, starting minimal
(inputs wired straight to outputs, no hidden nodes). Config: evo/neat_config.ini.
Parallel evaluation: pool.map inside our own fitness function (see run()).
"""

import argparse
import copy
import csv
import json
import pickle
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import neat
import numpy as np

from env.game import run_episode
from env.track import load_track
from evo.ga_deap import TEST_SEEDS, TEST_TRACKS, TRAIN_TRACKS, VALIDATION_SEEDS

CONFIG_PATH = "evo/neat_config.ini"


def make_config(path: str = CONFIG_PATH, pop_size: int | None = None) -> neat.Config:
    """Load a neat-python config.

    Args:
        path: path to the NEAT .ini config.
        pop_size: overrides [NEAT] pop_size if given (for ablations).

    Returns:
        neat.Config with DefaultGenome, DefaultReproduction,
        DefaultSpeciesSet, DefaultStagnation.
    """
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, path)
    if config.genome_config.num_inputs != 8 or config.genome_config.num_outputs != 2:
        raise ValueError("NEAT config must have num_inputs = 8 and num_outputs = 2")
    if pop_size is not None:
        config.pop_size = pop_size
    return config


def make_policy(genome, config: neat.Config):
    """Wrap a NEAT genome as a policy callable.

    Args:
        genome: neat.DefaultGenome.
        config: neat.Config.

    Returns:
        Callable obs (np.ndarray (8,)) -> (steer, throttle), each in [-1, 1].
        Builds the network once, not every frame.
    """
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    def policy(obs):
        steer, throttle = net.activate(obs)
        return float(steer), float(throttle)
    return policy


def evaluate(genome, config: neat.Config, track_names: tuple[str, ...], seed: int) -> float:
    """Fitness of one genome. Top-level so pool.map can pickle it.

    Args:
        genome: neat.DefaultGenome.
        config: neat.Config.
        track_names: tracks to drive; one episode each.
        seed: generation's shared episode seed.

    Returns:
        Mean of per-track env.game fitness (same aggregation as ga_deap.evaluate).
    """
    policy = make_policy(genome, config)
    scores = [run_episode(policy, load_track(name), seed)["fitness"] for name in track_names]
    return sum(scores) / len(scores)


def network_size(genome, config: neat.Config) -> tuple[int, int]:
    """(hidden nodes, enabled connections) of a genome."""
    hidden = sum(k not in config.genome_config.output_keys for k in genome.nodes)
    return hidden, sum(c.enabled for c in genome.connections.values())


def run(config: neat.Config, seed: int, generations: int = 100,
        track_names: tuple[str, ...] = TRAIN_TRACKS, pool=None, verbose: bool = False) -> dict:
    """One full NEAT run.

    Args:
        config: neat.Config.
        seed: master RNG seed (neat-python uses `random`); also derives
            per-generation episode seeds.
        generations: number of generations.
        track_names: training tracks.
        pool: multiprocessing.Pool or None (serial).
        verbose: print each generation's stats line.

    Returns:
        {"best_genome": neat.DefaultGenome, "best_fitness": float,
         "champion_genome": neat.DefaultGenome, "champion_fitness": float, "champion_gen": int,
         "logbook": list[dict]  # per-gen min/avg/max fitness, species, best net size,
         "evals": int}

    Same protocol as ga_deap.run: every generation draws a new episode seed and
    scores the whole population on it (neat-python re-evaluates elites too).
    neat.ParallelEvaluator can't pass a shared seed, so the fitness function
    below maps `evaluate` over the pool itself. Champion = the generation best
    with the highest mean fitness over VALIDATION_SEEDS.
    """
    seed_rng = np.random.default_rng(seed)   # episode seeds: own stream, as in the GA
    pop = neat.Population(config, seed=seed)
    mapper = pool.map if pool is not None else map
    logbook, gen_bests = [], []

    def fitness_function(genomes, config):
        episode_seed = int(seed_rng.integers(2**31))
        fits = list(mapper(partial(evaluate, config=config, track_names=track_names, seed=episode_seed),
                           [g for _, g in genomes]))
        for (_, g), fit in zip(genomes, fits):
            g.fitness = fit
        best = max((g for _, g in genomes), key=lambda g: g.fitness)
        gen_bests.append(copy.deepcopy(best))
        hidden, conns = network_size(best, config)
        rec = {"gen": len(logbook), "evals": len(genomes), "min": min(fits), "avg": float(np.mean(fits)),
               "max": max(fits), "species": len(pop.species.species), "hidden": hidden, "conns": conns}
        logbook.append(rec)
        if verbose:
            print(f"gen {rec['gen']:3d}  min {rec['min']:7.1f}  avg {rec['avg']:7.1f}  max {rec['max']:7.1f}  "
                  f"species {rec['species']:2d}  best net: {hidden} hidden, {conns} conns", flush=True)

    best = pop.run(fitness_function, generations)

    # Champion: re-score every generation's best on the same fixed seeds.
    mean_fit = np.zeros(len(gen_bests))
    for vseed in VALIDATION_SEEDS:
        fits = mapper(partial(evaluate, config=config, track_names=track_names, seed=vseed), gen_bests)
        mean_fit += np.array(list(fits)) / len(VALIDATION_SEEDS)
    champ = int(np.argmax(mean_fit))

    return {"best_genome": best, "best_fitness": float(best.fitness),
            "champion_genome": gen_bests[champ], "champion_fitness": float(mean_fit[champ]),
            "champion_gen": champ, "logbook": logbook, "evals": sum(r["evals"] for r in logbook)}


def track_report(genome, config: neat.Config, track_names: tuple[str, ...], seeds=TEST_SEEDS) -> dict:
    """Laps and crashes per track over fixed seeds; same format as ga_deap.track_report.

    Returns:
        {track: {"laps": mean laps per episode, "crashes": wall endings out of len(seeds),
                 "fitness": mean fitness}}
    """
    policy = make_policy(genome, config)
    out = {}
    for name in track_names:
        track = load_track(name)
        res = [run_episode(policy, track, s) for s in seeds]
        out[name] = {"laps": float(np.mean([r["checkpoints_passed"] for r in res]) / len(track.checkpoints)),
                     "crashes": sum(r["reason"] == "wall" for r in res),
                     "fitness": float(np.mean([r["fitness"] for r in res]))}
    return out


def save_run(out_dir: Path, config: neat.Config, args: dict, result: dict, report: dict,
             seconds: float) -> None:
    """Write neat_config.ini, config.json, logbook.csv, champion.{pkl,json} and summary.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    config.save(str(out_dir / "neat_config.ini"))   # resolved config, incl. --pop-size override
    (out_dir / "config.json").write_text(json.dumps(args, indent=2))
    with open(out_dir / "logbook.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(result["logbook"][0]))
        w.writeheader()
        w.writerows(result["logbook"])
    champ = result["champion_genome"]
    (out_dir / "champion.pkl").write_bytes(pickle.dumps(champ))
    hidden, conns = network_size(champ, config)
    (out_dir / "champion.json").write_text(json.dumps({
        "method": "neat",
        "genome_file": "champion.pkl",
        "config_file": "neat_config.ini",
        "validation_fitness": result["champion_fitness"],
        "generation": result["champion_gen"],
        "hidden_nodes": hidden,
        "enabled_connections": conns,
        "connections": [[i, o, c.weight] for (i, o), c in sorted(champ.connections.items()) if c.enabled],
    }, indent=2))
    (out_dir / "summary.json").write_text(json.dumps({
        "seed": args["seed"], "evals": result["evals"], "seconds": round(seconds, 1),
        "best_single_seed_fitness": result["best_fitness"],
        "champion_validation_fitness": result["champion_fitness"],
        "champion_generation": result["champion_gen"],
        "champion_hidden_nodes": hidden, "champion_connections": conns,
        "test_seeds": list(TEST_SEEDS), "tracks": report}, indent=2))


def load_champion(path: str | Path):
    """Policy from a champion.json written by save_run (genome + config read from its folder)."""
    path = Path(path)
    meta = json.loads(path.read_text())
    config = make_config(str(path.parent / meta["config_file"]))
    genome = pickle.loads((path.parent / meta["genome_file"]).read_bytes())
    return make_policy(genome, config)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: parse config path, seed, generations; run; save results/.

    Example: python -m evo.neat_run --seed 0 --generations 100 --out results/neat/default
    Writes <out>/seed<seed>/{config,champion,summary}.json + champion.pkl,
    logbook.csv, neat_config.ini.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--config", default=CONFIG_PATH, help="neat-python .ini")
    parser.add_argument("--pop-size", type=int, default=None, help="override config pop_size")
    parser.add_argument("--processes", type=int, default=None, help="pool workers (None = all cores, 1 = no pool)")
    parser.add_argument("--out", default="results/neat/default", help="run folder parent")
    parser.add_argument("--train-tracks", default=",".join(TRAIN_TRACKS), help="comma-separated")
    parser.add_argument("--test-tracks", default=",".join(TEST_TRACKS), help="comma-separated, report only")
    args = parser.parse_args(argv)
    config = make_config(args.config, args.pop_size)
    train_tracks = tuple(args.train_tracks.split(","))
    test_tracks = tuple(args.test_tracks.split(","))

    t0 = time.perf_counter()
    if args.processes == 1:
        result = run(config, args.seed, args.generations, train_tracks, verbose=True)
    else:
        with Pool(args.processes) as pool:
            result = run(config, args.seed, args.generations, train_tracks, pool, verbose=True)
    seconds = time.perf_counter() - t0
    report = track_report(result["champion_genome"], config, train_tracks + test_tracks)

    out_dir = Path(args.out) / f"seed{args.seed}"
    save_run(out_dir, config, {**vars(args), "pop_size": config.pop_size,
                               "train_tracks": train_tracks, "test_tracks": test_tracks},
             result, report, seconds)
    hidden, conns = network_size(result["champion_genome"], config)
    print(f"\n{result['evals']} evals in {seconds:.0f} s. Champion from gen {result['champion_gen']}, "
          f"validation fitness {result['champion_fitness']:.0f}, network {hidden} hidden / {conns} conns.")
    print(f"Champion on {len(TEST_SEEDS)} test seeds:")
    for name, r in report.items():
        tag = "held out" if name in test_tracks else "train"
        print(f"  {name:13s} {tag:8s} laps {r['laps']:.2f}   crashes {r['crashes']}/{len(TEST_SEEDS)}")
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
