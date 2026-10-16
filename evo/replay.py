"""Watch saved drivers in the pygame window.

One champion:      python -m evo.replay results/ga/default/seed0/champion.json [--track peanut] [--seed 3]
Several (a race):  python -m evo.replay results/ga/default/seed*/champion.json --track wavy
    -> all start together; the fastest is drawn in colour, the rest as gray ghosts.

--frames DIR saves every frame as PNG (for videos) instead of playing at 60 fps.
"""

import argparse
import json
from pathlib import Path

from env.render import race, replay
from evo.ga_deap import make_policy


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Replay saved GA champions.")
    parser.add_argument("champions", nargs="+", help="champion.json file(s) written by evo.ga_deap")
    parser.add_argument("--track", default="peanut")
    parser.add_argument("--seed", type=int, default=None, help="episode seed (default: canonical start)")
    parser.add_argument("--frames", default=None, help="folder to save PNG frames")
    args = parser.parse_args(argv)

    policies = [make_policy(json.loads(Path(p).read_text())["genome"]) for p in args.champions]
    if len(policies) == 1:
        replay(policies[0], args.track, args.seed, args.frames)
    else:
        labels = [Path(p).parent.name or Path(p).stem for p in args.champions]   # e.g. "seed3"
        race(policies, labels, args.track, args.seed, args.frames)


if __name__ == "__main__":
    main()
