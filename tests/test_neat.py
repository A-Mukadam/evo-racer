"""Tests for NEAT (evo/neat_run.py)."""

import numpy as np
import pytest

from env.game import run_episode
from env.track import load_track
from evo.neat_run import evaluate, make_config, make_policy, network_size, run


@pytest.fixture(scope="module")
def config():
    return make_config()


def _genome(config, key=1):
    g = config.genome_type(key)
    g.configure_new(config.genome_config)
    return g


def test_config_starts_minimal(config):
    """8 inputs wired straight to 2 outputs: no hidden nodes, 16 connections."""
    assert config.genome_config.num_inputs == 8 and config.genome_config.num_outputs == 2
    import neat
    neat.Population(config, seed=0)   # sets up the innovation tracker configure_new needs
    assert network_size(_genome(config), config) == (0, 16)


def test_policy_outputs_in_range_and_python_floats(config):
    import neat
    neat.Population(config, seed=0)
    policy = make_policy(_genome(config), config)
    rng = np.random.default_rng(0)
    for _ in range(100):
        steer, throttle = policy(rng.uniform(0, 1, 8))
        assert type(steer) is float and type(throttle) is float
        assert -1.0 <= steer <= 1.0 and -1.0 <= throttle <= 1.0


def test_evaluate_is_mean_over_tracks(config):
    import pickle
    from functools import partial
    import neat
    from evo.ga_deap import TRAIN_TRACKS

    neat.Population(config, seed=0)
    g = _genome(config)
    fit = evaluate(g, config, TRAIN_TRACKS, seed=7)
    per_track = [run_episode(make_policy(g, config), load_track(t), 7)["fitness"] for t in TRAIN_TRACKS]
    assert fit == pytest.approx(sum(per_track) / 3)
    assert evaluate(g, config, TRAIN_TRACKS, seed=7) == fit                      # deterministic
    pickle.dumps(partial(evaluate, config=config, track_names=TRAIN_TRACKS, seed=7))   # pool.map
    pickle.dumps(g)


def test_neat_run_small():
    """A tiny run: right bookkeeping, deterministic given the seed."""
    cfg = make_config(pop_size=10)
    a, b = run(cfg, seed=0, generations=3), run(make_config(pop_size=10), seed=0, generations=3)
    assert a["evals"] == 10 * 3 and len(a["logbook"]) == 3
    assert a["best_fitness"] == max(r["max"] for r in a["logbook"])
    assert [r["max"] for r in a["logbook"]] == [r["max"] for r in b["logbook"]]
    assert a["champion_gen"] in (0, 1, 2)


def test_cli_writes_results(tmp_path):
    """End to end, serial, tiny: the CLI saves a champion that loads back."""
    import json
    from evo.champion import load_policy
    from evo.neat_run import main

    main(["--pop-size", "6", "--generations", "2", "--processes", "1", "--seed", "3",
          "--out", str(tmp_path)])
    run_dir = tmp_path / "seed3"
    assert {p.name for p in run_dir.iterdir()} == {"config.json", "logbook.csv", "champion.json",
                                                    "champion.pkl", "neat_config.ini", "summary.json"}
    champ = json.loads((run_dir / "champion.json").read_text())
    assert champ["method"] == "neat" and champ["generation"] in (0, 1)
    summary = json.loads((run_dir / "summary.json").read_text())
    assert set(summary["tracks"]) == {"oval", "rounded_rect", "wavy", "peanut"}
    assert json.loads((run_dir / "config.json").read_text())["pop_size"] == 6
    steer, throttle = load_policy(run_dir / "champion.json")(np.full(8, 0.5))
    assert -1.0 <= steer <= 1.0 and -1.0 <= throttle <= 1.0
