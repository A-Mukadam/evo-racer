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
simple_tracks.py          parametric tracks + F1-inspired piece-built courses -> (walls, checkpoints, centerline)
env/track.py              wraps simple_tracks, builds spatial grid
env/car.py                arcade physics + 7 raycast sensors
env/render.py             pygame drawing + ray debug overlay (never used during evolution)
env/game.py               RaceEnv: reset(track, seed) / step(steer, throttle), fitness
evo/ga_deap.py            fixed-topology GA (8-8-2 tanh net, 90-float genome)
evo/replay.py             watch a saved champion in the pygame window
evo/neat_run.py           NEAT
experiments/run_sweep.py  30-seed sweeps, baselines (hand-coded heuristic_policy), held-out eval, ablations
experiments/evaluate.py   score saved champions + the hand-coded baseline on any tracks
tests/test_geometry.py    raycast / grid / collision / checkpoint tests
docs/project_notes.md     plain-language project notes for the presentation (images in docs/img/)
```

## Commands

```bash
python simple_tracks.py            # preview tracks -> tracks_preview.png
python -m env.render [track]       # W1: manual drive with ray overlay (box, oval, rounded_rect, wavy, peanut)
pytest                             # geometry + env tests
```

Manual drive keys: arrows/WASD drive, R restart, N random spawn, TAB next track,
C toggle checkpoints, ESC quit. `box` is a 4-wall sanity box for checking ray distances.

GA (W2): one run = 100 x 100 generations, ~80 s on 12 cores. Writes
`results/ga/default/seed0/` (config, logbook.csv, champion.json, summary.json)
and prints the champion's laps and crashes per track. `--help` lists every GAConfig option.

```bash
python -m evo.ga_deap --seed 0
python -m evo.replay results/ga/default/seed0/champion.json --track peanut
python -m evo.replay results/ga/default/seed*/champion.json --track wavy   # race: fastest in colour, rest gray
```

F1 experiment: 4 F1-inspired courses (`f1_chicanes`, `f1_hairpins`, `f1_esses`,
`f1_reverse`). Train on 3, hold out `f1_hairpins`; score any champions on any tracks:

```bash
python -m evo.ga_deap --seed 0 --out results/ga/f1 --train-tracks f1_chicanes,f1_esses,f1_reverse --test-tracks f1_hairpins,peanut
python -m experiments.evaluate results/ga/f1/seed*/champion.json --tracks f1_hairpins,peanut
```

Not implemented yet (W3+):

```bash
python -m evo.neat_run
python -m experiments.run_sweep
```
