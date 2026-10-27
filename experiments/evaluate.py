"""Score saved champions (and the hand-coded baseline) on any tracks.

    python -m experiments.evaluate results/ga/default/seed*/champion.json --tracks f1_chicanes,f1_hairpins
    python -m experiments.evaluate results/neat/default/seed*/champion.json --tracks oval,rounded_rect,wavy,peanut

Prints mean laps per episode and crashes per track over TEST_SEEDS: one row per
champion, a mean +/- std row across champions, and a hand-coded baseline row.
Laps (checkpoints passed / checkpoints per lap) are the generalization-gap unit.
"""

import argparse
from pathlib import Path

import numpy as np

from env.game import run_episode
from env.track import load_track
from evo.champion import champion_labels, load_policy
from evo.ga_deap import TEST_SEEDS
from experiments.run_sweep import heuristic_policy


def score(policy, track_names, seeds=TEST_SEEDS) -> dict:
    """{track: (mean laps per episode, crashes out of len(seeds))}."""
    out = {}
    for name in track_names:
        track = load_track(name)
        res = [run_episode(policy, track, s) for s in seeds]
        out[name] = (float(np.mean([r["checkpoints_passed"] for r in res])) / len(track.checkpoints),
                     sum(r["reason"] == "wall" for r in res))
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score champions on tracks.")
    parser.add_argument("champions", nargs="+", help="champion.json files")
    parser.add_argument("--tracks", required=True, help="comma-separated track names")
    args = parser.parse_args(argv)
    tracks = args.tracks.split(",")
    n = len(TEST_SEEDS)

    rows = {label: score(load_policy(p), tracks)
            for label, p in zip(champion_labels(args.champions), args.champions)}
    width = max(16, *(len(t) + 4 for t in tracks))
    print(f"laps per episode (crashes out of {n}), {n} test seeds\n")
    print(f"{'':12s}" + "".join(f"{t:>{width}s}" for t in tracks))
    for label, r in rows.items():
        print(f"{label:12s}" + "".join(f"{f'{r[t][0]:.2f} ({r[t][1]})':>{width}s}" for t in tracks))
    laps = np.array([[r[t][0] for t in tracks] for r in rows.values()])
    crashes = np.array([[r[t][1] for t in tracks] for r in rows.values()])
    print(f"{'mean +/- sd':12s}" + "".join(f"{f'{m:.2f} +/- {s:.2f}':>{width}s}" for m, s in zip(laps.mean(0), laps.std(0))))
    print(f"{'crash rate':12s}" + "".join(f"{f'{c / (n * len(rows)):.0%}':>{width}s}" for c in crashes.sum(0)))
    h = score(heuristic_policy, tracks)
    print(f"{'hand-coded':12s}" + "".join(f"{f'{h[t][0]:.2f} ({h[t][1]})':>{width}s}" for t in tracks))


if __name__ == "__main__":
    main()
