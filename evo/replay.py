"""Watch saved drivers in the pygame window.

One champion:      python -m evo.replay results/ga/default/seed0/champion.json [--track peanut] [--seed 3]
NEAT works too:    python -m evo.replay results/neat/default/seed0/champion.json --track peanut
Several (a race):  python -m evo.replay results/ga/default/seed*/champion.json --track wavy
    -> all start together; the fastest is drawn in colour, the rest as gray ghosts.

--frames DIR saves every frame as PNG (for videos) instead of playing at 60 fps.
"""

import argparse
from pathlib import Path

from env.render import race, replay
from evo.champion import champion_labels, load_policy


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Replay saved GA / NEAT champions.")
    parser.add_argument("champions", nargs="+", help="champion.json file(s) written by evo.ga_deap or evo.neat_run")
    parser.add_argument("--track", default="peanut")
    parser.add_argument("--seed", type=int, default=None, help="episode seed (default: canonical start)")
    parser.add_argument("--frames", default=None, help="folder to save PNG frames")
    args = parser.parse_args(argv)

    policies = [load_policy(p) for p in args.champions]
    if len(policies) == 1:
        replay(policies[0], args.track, args.seed, args.frames)
    else:
        labels = champion_labels(args.champions)   # e.g. "seed3", or "neat/seed0" vs "ga/seed0"
        race(policies, labels, args.track, args.seed, args.frames)


if __name__ == "__main__":
    main()
