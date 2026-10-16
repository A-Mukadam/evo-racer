"""Tests for the fixed-topology GA (evo/ga_deap.py)."""

import numpy as np
import pytest

from env.game import run_episode
from env.track import load_track
from evo.ga_deap import GENOME_LEN, decode, forward, make_policy


def test_genome_length():
    assert GENOME_LEN == 90


def test_decode_layout():
    """W1 (8x8, row = input), then b1, W2 (8x2, row = hidden unit), then b2."""
    W1, b1, W2, b2 = decode(list(range(GENOME_LEN)))
    assert W1.shape == (8, 8) and b1.shape == (8,) and W2.shape == (8, 2) and b2.shape == (2,)
    assert W1[0, 0] == 0 and W1[0, 7] == 7 and W1[7, 7] == 63
    assert list(b1) == list(range(64, 72))
    assert W2[0, 0] == 72 and W2[0, 1] == 73 and W2[7, 1] == 87
    assert list(b2) == [88, 89]


def test_decode_rejects_wrong_length():
    with pytest.raises(ValueError):
        decode([0.0] * 89)


def test_forward_matches_hand_calculation():
    """Front sensor -> hidden 0 (weight 1); hidden 0 -> throttle (weight 2)."""
    g = np.zeros(GENOME_LEN)
    g[3 * 8 + 0] = 1.0          # W1[3, 0]: input 3 (front ray) -> hidden 0
    g[72 + 0 * 2 + 1] = 2.0     # W2[0, 1]: hidden 0 -> throttle
    obs = np.zeros(8)
    obs[3] = 0.5
    steer, throttle = forward(decode(g), obs)
    assert steer == 0.0
    assert throttle == pytest.approx(np.tanh(2.0 * np.tanh(0.5)))


def test_outputs_in_range_and_python_floats():
    rng = np.random.default_rng(0)
    for _ in range(100):
        params = decode(rng.normal(0, 5, GENOME_LEN))   # large weights saturate tanh
        steer, throttle = forward(params, rng.uniform(0, 1, 8))
        assert type(steer) is float and type(throttle) is float
        assert -1.0 <= steer <= 1.0 and -1.0 <= throttle <= 1.0


def test_policy_drives_the_env():
    """All-zero weights output (0, 0): the car never moves and the episode stalls."""
    res = run_episode(make_policy([0.0] * GENOME_LEN), load_track("oval"), None)
    assert res["reason"] == "stall" and res["checkpoints_passed"] == 0


def test_evaluate_is_mean_over_tracks():
    import pickle
    from functools import partial
    from evo.ga_deap import TRAIN_TRACKS, evaluate

    genome = np.random.default_rng(1).uniform(-1, 1, GENOME_LEN).tolist()
    (fit,) = evaluate(genome, TRAIN_TRACKS, seed=7)
    per_track = [run_episode(make_policy(genome), load_track(t), 7)["fitness"] for t in TRAIN_TRACKS]
    assert fit == pytest.approx(sum(per_track) / 3)
    assert evaluate(genome, TRAIN_TRACKS, seed=7) == (fit,)               # deterministic
    pickle.dumps(partial(evaluate, track_names=TRAIN_TRACKS, seed=7))    # pool.map needs this


def test_ga_run_small():
    """A tiny run: right bookkeeping, deterministic given the seed."""
    from evo.ga_deap import GAConfig, run

    cfg = GAConfig(pop_size=8, generations=3)
    a, b = run(cfg, seed=0), run(cfg, seed=0)
    assert a["evals"] == 8 * 3
    assert len(a["logbook"]) == 3 and len(a["best_genome"]) == GENOME_LEN
    assert a["best_fitness"] == max(a["logbook"].select("max"))
    assert a["logbook"].select("max") == b["logbook"].select("max")
    assert a["best_genome"] == b["best_genome"]


def test_individual_pickles():
    import pickle
    from deap import creator

    ind = creator.Individual([0.5] * GENOME_LEN)
    ind.fitness.values = (3.0,)
    back = pickle.loads(pickle.dumps(ind))
    assert back == ind and back.fitness.values == (3.0,)


def test_cli_writes_results(tmp_path):
    """End to end, serial, tiny: the CLI saves a champion that loads back."""
    import json
    from evo.ga_deap import main

    main(["--pop-size", "6", "--generations", "2", "--processes", "1", "--seed", "3",
          "--out", str(tmp_path)])
    run_dir = tmp_path / "seed3"
    assert {p.name for p in run_dir.iterdir()} == {"config.json", "logbook.csv", "champion.json", "summary.json"}
    champ = json.loads((run_dir / "champion.json").read_text())
    assert len(champ["genome"]) == GENOME_LEN and champ["generation"] in (0, 1)
    summary = json.loads((run_dir / "summary.json").read_text())
    assert set(summary["tracks"]) == {"oval", "rounded_rect", "wavy", "peanut"}
    assert json.loads((run_dir / "config.json").read_text())["pop_size"] == 6
