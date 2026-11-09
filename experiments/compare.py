"""Compare groups of saved runs: e.g. seeds 0-4 vs 5-29, GA vs NEAT, fixed vs adaptive sigma.

Each group is LABEL=RUN_DIR[:FIRST-LAST], one champion (summary.json) per seed folder.
    python -m experiments.compare "GA 0-4=results/ga/default:0-4" "GA 5-29=results/ga/default:5-29"
Prints per-group results on the held-out track, then tests every group against the first:
a permutation test on mean held-out laps and Fisher's exact test on how many seeds
lap it with no crashes. Both are two-sided and assume nothing about distributions.
"""

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np

HELD_OUT = "peanut"


def load_group(spec: str) -> tuple[str, list[dict]]:
    """Parse LABEL=DIR[:A-B] and load each seed's summary.json (missing seeds are skipped)."""
    label, _, where = spec.partition("=")
    folder, _, rng = where.partition(":")
    if rng:
        a, b = (int(x) for x in rng.split("-"))
        paths = [Path(folder) / f"seed{s}" / "summary.json" for s in range(a, b + 1)]
    else:
        paths = sorted(Path(folder).glob("seed*/summary.json"), key=lambda p: int(p.parent.name[4:]))
    return label, [json.loads(p.read_text()) for p in paths if p.exists()]


def seed_stats(summary: dict, held_out: str = HELD_OUT) -> dict:
    """One champion: held-out laps/crashes and training-track laps/crashes."""
    t = summary["tracks"]
    train = [k for k in t if k != held_out]
    n_eps = len(summary["test_seeds"])
    return {"laps": t[held_out]["laps"], "crashes": t[held_out]["crashes"],
            "clean": t[held_out]["crashes"] == 0 and t[held_out]["laps"] >= 1,
            "train_laps": float(np.mean([t[k]["laps"] for k in train])),
            "train_crashes": sum(t[k]["crashes"] for k in train), "train_eps": n_eps * len(train),
            "eps": n_eps, "minutes": summary["seconds"] / 60, "gen": summary["champion_generation"]}


def permutation_p(a, b, n: int = 20_000, seed: int = 0) -> float:
    """Two-sided p-value for a difference in means, by shuffling group labels."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    both, k = np.concatenate([a, b]), len(a)
    obs = abs(a.mean() - b.mean())
    rng = np.random.default_rng(seed)
    hits = sum(abs(p[:k].mean() - p[k:].mean()) >= obs - 1e-12
               for p in (rng.permutation(both) for _ in range(n)))
    return (hits + 1) / (n + 1)


def fisher_p(x1: int, n1: int, x2: int, n2: int) -> float:
    """Two-sided Fisher's exact test: x1 of n1 vs x2 of n2 successes."""
    total, k = n1 + n2, x1 + x2
    prob = lambda x: comb(n1, x) * comb(n2, k - x) / comb(total, k)
    p_obs = prob(x1)
    return min(1.0, sum(prob(x) for x in range(max(0, k - n2), min(k, n1) + 1) if prob(x) <= p_obs * (1 + 1e-9)))


def describe(label: str, runs: list[dict]) -> dict:
    s = [seed_stats(r) for r in runs]
    laps = [x["laps"] for x in s]
    return {"label": label, "n": len(s), "laps": laps,
            "laps_mean": float(np.mean(laps)), "laps_sd": float(np.std(laps, ddof=1)) if len(s) > 1 else 0.0,
            "clean": sum(x["clean"] for x in s),
            "crash_rate": sum(x["crashes"] for x in s) / sum(x["eps"] for x in s),
            "train_laps": float(np.mean([x["train_laps"] for x in s])),
            "train_crash_rate": sum(x["train_crashes"] for x in s) / sum(x["train_eps"] for x in s),
            "minutes": float(np.mean([x["minutes"] for x in s])),
            "gen": float(np.mean([x["gen"] for x in s]))}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("groups", nargs="+", help='LABEL=RUN_DIR[:FIRST-LAST]')
    parser.add_argument("--json", default=None, help="also write the numbers here")
    args = parser.parse_args(argv)
    rows = [describe(*load_group(g)) for g in args.groups]

    print(f"{'group':16s} {'seeds':>5s} {'peanut laps':>13s} {'clean':>7s} {'crash%':>7s} "
          f"{'train laps':>10s} {'train crash%':>12s} {'min/run':>8s} {'champ gen':>9s}")
    for r in rows:
        print(f"{r['label']:16s} {r['n']:5d} {r['laps_mean']:6.2f} ± {r['laps_sd']:4.2f} "
              f"{r['clean']:3d}/{r['n']:<3d} {100 * r['crash_rate']:6.0f}% {r['train_laps']:10.2f} "
              f"{100 * r['train_crash_rate']:11.1f}% {r['minutes']:8.1f} {r['gen']:9.0f}")
    base = rows[0]
    tests = []
    for r in rows[1:]:
        t = {"a": base["label"], "b": r["label"],
             "laps_p": permutation_p(base["laps"], r["laps"]),
             "clean_p": fisher_p(base["clean"], base["n"], r["clean"], r["n"])}
        tests.append(t)
        print(f"\n{r['label']} vs {base['label']}: peanut laps {r['laps_mean'] - base['laps_mean']:+.2f} "
              f"(permutation p = {t['laps_p']:.2f}); clean seeds {r['clean']}/{r['n']} vs "
              f"{base['clean']}/{base['n']} (Fisher p = {t['clean_p']:.2f})")
    if args.json:
        Path(args.json).write_text(json.dumps({"groups": rows, "tests": tests}, indent=2))


if __name__ == "__main__":
    main()
