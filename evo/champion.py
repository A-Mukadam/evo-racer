"""Load any saved champion.json (GA or NEAT) as a policy callable."""

import json
from pathlib import Path


def load_policy(path: str | Path):
    """Policy obs (np.ndarray (8,)) -> (steer, throttle) from a champion.json.

    GA champions store the 90-float genome inline; NEAT champions are marked
    "method": "neat" and point at a pickled genome + config in the same folder.
    """
    meta = json.loads(Path(path).read_text())
    if meta.get("method") == "neat":
        from evo.neat_run import load_champion
        return load_champion(path)
    from evo.ga_deap import make_policy
    return make_policy(meta["genome"])


def champion_method(path: str | Path) -> str:
    """"neat" or "ga": which method wrote this champion.json."""
    return json.loads(Path(path).read_text()).get("method", "ga")


def champion_labels(paths) -> list[str]:
    """Short unique names for champion.json paths (results/<method>/<run>/<seed>/):
    the seed folder, prefixed by the method and/or run folder whenever those
    differ between the paths given: "seed3"; "neat/seed0" vs "ga/seed0";
    "default/seed0" vs "f1/seed0"; "ga/f1/seed3" vs "neat/default/seed0".
    Callers key results by these labels, so they must not collide."""
    paths = [Path(p) for p in paths]

    def up(p: Path, i: int) -> str:   # name of the i-th parent folder, "" past the top
        return p.parents[i].name if len(p.parents) > i else ""

    show_method = len({up(p, 2) for p in paths}) > 1
    show_run = len({up(p, 1) for p in paths}) > 1
    labels = ["/".join([up(p, 2)] * show_method + [up(p, 1)] * show_run + [up(p, 0) or p.stem])
              for p in paths]
    return labels if len(set(labels)) == len(labels) else [str(p.parent) for p in paths]
