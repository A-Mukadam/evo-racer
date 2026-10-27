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


def champion_labels(paths) -> list[str]:
    """Short names for champion.json paths: the run folder ("seed3"), or
    "<method>/seed3" (e.g. "neat/seed0") when run folders collide."""
    paths = [Path(p) for p in paths]
    labels = [p.parent.name or p.stem for p in paths]
    if len(set(labels)) < len(labels):
        labels = [f"{p.parents[2].name}/{p.parent.name}" if len(p.parents) > 2 else str(p.parent) for p in paths]
    return labels
