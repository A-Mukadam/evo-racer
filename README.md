# evo-racer
Evolving neural network drivers for a 2D racing simulator using genetic algorithms and NEAT.

Train on 3 tracks, test on a 4th unseen track, report the generalization gap.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run everything from the repo root (modules import `simple_tracks` and `env.*` as top-level).

## Layout

```
simple_tracks.py          parametric tracks -> (walls, checkpoints, centerline)
env/track.py              wraps simple_tracks, builds spatial grid
env/car.py                arcade physics + 7 raycast sensors
env/render.py             pygame drawing + ray debug overlay (never used during evolution)
env/game.py               RaceEnv: reset(track, seed) / step(steer, throttle), fitness
evo/ga_deap.py            fixed-topology GA (8-8-2 tanh net, 90-float genome)
evo/neat_run.py           NEAT
experiments/run_sweep.py  30-seed sweeps, baselines, held-out eval, ablations
tests/test_geometry.py    raycast / grid / collision / checkpoint tests
```

## Commands (once implemented)

```bash
python simple_tracks.py            # preview tracks -> tracks_preview.png
python -m env.render               # W1: manual drive with ray overlay
pytest tests/
python -m evo.ga_deap
python -m evo.neat_run
python -m experiments.run_sweep
```
