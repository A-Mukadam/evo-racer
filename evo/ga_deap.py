"""Fixed-topology neuroevolution with DEAP.

Network: 8 inputs -> 8 hidden (tanh) -> 2 outputs (steer, throttle).
Genome: flat list of 90 floats = W1 (8x8) + b1 (8) + W2 (8x2) + b2 (2).
Parallel evaluation: toolbox.register("map", pool.map).
"""

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass, fields
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from deap import algorithms, base, creator, tools

from env.game import run_episode
from env.track import load_track

N_INPUTS = 8
N_HIDDEN = 8
N_OUTPUTS = 2
GENOME_LEN = N_INPUTS * N_HIDDEN + N_HIDDEN + N_HIDDEN * N_OUTPUTS + N_OUTPUTS  # 90

TRAIN_TRACKS = ("oval", "rounded_rect", "wavy")
TEST_TRACKS = ("peanut",)
INIT_RANGE = 1.0   # initial weights ~ uniform(-INIT_RANGE, INIT_RANGE)
VALIDATION_SEEDS = tuple(range(10_000, 10_005))  # champion pick, training tracks
TEST_SEEDS = tuple(range(20_000, 20_010))        # final report, all tracks

# Created at import, not in build_toolbox: pool workers (spawned fresh on macOS)
# import this module and need these classes to unpickle Individuals.
# Guarded so re-imports and repeated runs don't re-register them.
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)


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
        selection: "tournament" | "roulette".
        tournsize: tournament size, if selection == "tournament".
        elitism: number of top individuals copied unchanged, >= 0.
        train_tracks: track names evaluated for fitness.
        processes: worker processes for multiprocessing.Pool (None = all cores,
            1 = no pool).
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
        W1[i, j] = weight from input i to hidden unit j; W2[j, k] = hidden j to output k.
    """
    g = np.asarray(genome, dtype=float)
    if g.shape != (GENOME_LEN,):
        raise ValueError(f"genome must have {GENOME_LEN} floats, got shape {g.shape}")
    i = 0
    W1 = g[i:i + N_INPUTS * N_HIDDEN].reshape(N_INPUTS, N_HIDDEN)
    i += N_INPUTS * N_HIDDEN
    b1 = g[i:i + N_HIDDEN]
    i += N_HIDDEN
    W2 = g[i:i + N_HIDDEN * N_OUTPUTS].reshape(N_HIDDEN, N_OUTPUTS)
    i += N_HIDDEN * N_OUTPUTS
    b2 = g[i:i + N_OUTPUTS]
    return W1, b1, W2, b2


def forward(params, obs: np.ndarray) -> tuple[float, float]:
    """Network forward pass.

    Args:
        params: output of decode().
        obs: np.ndarray shape (8,), float in [0, 1].

    Returns:
        (steer, throttle), each in [-1, 1] (tanh outputs).
    """
    W1, b1, W2, b2 = params
    hidden = np.tanh(obs @ W1 + b1)
    steer, throttle = np.tanh(hidden @ W2 + b2)
    return float(steer), float(throttle)


def make_policy(genome):
    """Wrap a genome as a policy callable.

    Args:
        genome: sequence of GENOME_LEN floats.

    Returns:
        Callable obs (np.ndarray (8,)) -> (steer, throttle). Decodes once,
        not every frame.
    """
    params = decode(genome)
    return lambda obs: forward(params, obs)


def evaluate(genome, track_names: tuple[str, ...], seed: int) -> tuple[float]:
    """DEAP fitness function. Must be picklable (top-level) for pool.map.

    Args:
        genome: sequence of GENOME_LEN floats.
        track_names: tracks to drive; one episode each.
        seed: generation's shared episode seed.

    Returns:
        1-tuple (fitness,): mean of per-track env.game fitness.
    """
    policy = make_policy(genome)
    scores = [run_episode(policy, load_track(name), seed)["fitness"] for name in track_names]
    return (sum(scores) / len(scores),)


def build_toolbox(cfg: GAConfig, pool=None):
    """Register creator types and operators.

    Args:
        cfg: GAConfig.
        pool: multiprocessing.Pool or None (serial map).

    Returns:
        deap.base.Toolbox with attr, individual, population, evaluate, mate,
        mutate, select, and map registered. `evaluate` still needs seed=.
    """
    toolbox = base.Toolbox()
    toolbox.register("attr", random.uniform, -INIT_RANGE, INIT_RANGE)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr, GENOME_LEN)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate, track_names=cfg.train_tracks)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=cfg.sigma, indpb=cfg.indpb)
    if cfg.selection == "tournament":
        toolbox.register("select", tools.selTournament, tournsize=cfg.tournsize)
    elif cfg.selection == "roulette":
        toolbox.register("select", tools.selRoulette)
    else:
        raise ValueError(f"unknown selection {cfg.selection!r}")
    toolbox.register("map", pool.map if pool is not None else map)
    return toolbox


def run(cfg: GAConfig, seed: int, pool=None, verbose: bool = False) -> dict:
    """One full GA run.

    Args:
        cfg: GAConfig.
        seed: master RNG seed (random + numpy); also derives per-generation
            episode seeds.
        pool: multiprocessing.Pool or None.
        verbose: print each generation's stats line.

    Returns:
        {"best_genome": list[float], "best_fitness": float,
         "champion_genome": list[float], "champion_fitness": float, "champion_gen": int,
         "logbook": deap.tools.Logbook (per-gen min/mean/max fitness),
         "evals": int  # total fitness evaluations, for matched-budget baselines}

    Every generation draws a new episode seed and re-scores the WHOLE
    population on it, elites included, so all drivers in a generation are
    compared on the same start. (DEAP's eaSimple / eaMuPlusLambda keep old
    scores, which would compare drivers on different starts.)
    best_fitness is the best single-generation score, so it can be a lucky
    start. The champion is the generation best with the highest mean fitness
    over VALIDATION_SEEDS; report that one. (`evals` excludes the
    re-scoring: it counts the search budget only.)
    """
    random.seed(seed)                              # DEAP operators use `random`
    seed_rng = np.random.default_rng(seed)         # episode seeds: own stream, so they
    toolbox = build_toolbox(cfg, pool)             # don't depend on operator draws

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    logbook = tools.Logbook()
    logbook.header = ["gen", "evals", "min", "avg", "max"]

    pop = toolbox.population(n=cfg.pop_size)
    best_genome, best_fitness, evals = None, -np.inf, 0
    gen_bests = []
    for gen in range(cfg.generations):
        episode_seed = int(seed_rng.integers(2**31))
        fits = toolbox.map(partial(toolbox.evaluate, seed=episode_seed), pop)
        for ind, fit in zip(pop, fits):
            ind.fitness.values = fit
        evals += len(pop)
        logbook.record(gen=gen, evals=len(pop), **stats.compile(pop))
        if verbose:
            print(logbook.stream, flush=True)

        gen_best = tools.selBest(pop, 1)[0]
        gen_bests.append(list(gen_best))
        if gen_best.fitness.values[0] > best_fitness:
            best_genome, best_fitness = list(gen_best), gen_best.fitness.values[0]

        if gen < cfg.generations - 1:   # breed the next generation
            elites = [toolbox.clone(ind) for ind in tools.selBest(pop, cfg.elitism)]
            offspring = [toolbox.clone(ind) for ind in toolbox.select(pop, cfg.pop_size - cfg.elitism)]
            pop = elites + algorithms.varAnd(offspring, toolbox, cfg.cx_prob, cfg.mut_prob)

    # Champion: re-score every generation's best on the same fixed seeds.
    mean_fit = np.zeros(len(gen_bests))
    for vseed in VALIDATION_SEEDS:
        fits = toolbox.map(partial(toolbox.evaluate, seed=vseed), gen_bests)
        mean_fit += np.array([f[0] for f in fits]) / len(VALIDATION_SEEDS)
    champ = int(np.argmax(mean_fit))

    return {"best_genome": best_genome, "best_fitness": float(best_fitness),
            "champion_genome": gen_bests[champ], "champion_fitness": float(mean_fit[champ]),
            "champion_gen": champ, "logbook": logbook, "evals": evals}


def track_report(genome, track_names: tuple[str, ...], seeds=TEST_SEEDS) -> dict:
    """Laps and crashes per track over fixed seeds (lap fraction = generalization-gap unit).

    Returns:
        {track: {"laps": mean laps per episode, "crashes": wall endings out of len(seeds),
                 "fitness": mean fitness}}
    """
    policy = make_policy(genome)
    out = {}
    for name in track_names:
        track = load_track(name)
        res = [run_episode(policy, track, s) for s in seeds]
        out[name] = {"laps": float(np.mean([r["checkpoints_passed"] for r in res]) / len(track.checkpoints)),
                     "crashes": sum(r["reason"] == "wall" for r in res),
                     "fitness": float(np.mean([r["fitness"] for r in res]))}
    return out


def save_run(out_dir: Path, cfg: GAConfig, seed: int, result: dict, report: dict, seconds: float,
             test_tracks: tuple[str, ...] = TEST_TRACKS) -> None:
    """Write config.json, logbook.csv, champion.json and summary.json into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({**asdict(cfg), "test_tracks": test_tracks, "seed": seed}, indent=2))
    with open(out_dir / "logbook.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gen", "evals", "min", "avg", "max"])
        for rec in result["logbook"]:
            w.writerow([rec["gen"], rec["evals"], rec["min"], rec["avg"], rec["max"]])
    (out_dir / "champion.json").write_text(json.dumps({
        "genome": result["champion_genome"],
        "validation_fitness": result["champion_fitness"],
        "generation": result["champion_gen"]}, indent=2))
    (out_dir / "summary.json").write_text(json.dumps({
        "seed": seed, "evals": result["evals"], "seconds": round(seconds, 1),
        "best_single_seed_fitness": result["best_fitness"],
        "champion_validation_fitness": result["champion_fitness"],
        "champion_generation": result["champion_gen"],
        "test_seeds": list(TEST_SEEDS), "tracks": report}, indent=2))


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: parse GAConfig fields + seed, run, save results/.

    Example: python -m evo.ga_deap --seed 0 --generations 100 --out results/ga/default
    Writes <out>/seed<seed>/{config,champion,summary}.json + logbook.csv.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/ga/default", help="run folder parent")
    parser.add_argument("--train-tracks", default=",".join(TRAIN_TRACKS), help="comma-separated")
    parser.add_argument("--test-tracks", default=",".join(TEST_TRACKS), help="comma-separated, report only")
    for f in fields(GAConfig):
        if f.name != "train_tracks":
            kind = type(f.default) if f.default is not None else int
            parser.add_argument("--" + f.name.replace("_", "-"), type=kind, default=f.default)
    args = parser.parse_args(argv)
    cfg = GAConfig(**{f.name: getattr(args, f.name) for f in fields(GAConfig) if f.name != "train_tracks"},
                   train_tracks=tuple(args.train_tracks.split(",")))
    test_tracks = tuple(args.test_tracks.split(","))

    t0 = time.perf_counter()
    if cfg.processes == 1:
        result = run(cfg, args.seed, verbose=True)
    else:
        with Pool(cfg.processes) as pool:
            result = run(cfg, args.seed, pool, verbose=True)
    seconds = time.perf_counter() - t0
    report = track_report(result["champion_genome"], cfg.train_tracks + test_tracks)

    out_dir = Path(args.out) / f"seed{args.seed}"
    save_run(out_dir, cfg, args.seed, result, report, seconds, test_tracks)
    print(f"\n{result['evals']} evals in {seconds:.0f} s. Champion from gen {result['champion_gen']}, "
          f"validation fitness {result['champion_fitness']:.0f}.")
    print(f"Champion on {len(TEST_SEEDS)} test seeds:")
    for name, r in report.items():
        tag = "held out" if name in test_tracks else "train"
        print(f"  {name:13s} {tag:8s} laps {r['laps']:.2f}   crashes {r['crashes']}/{len(TEST_SEEDS)}")
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
