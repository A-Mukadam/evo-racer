# CLAUDE.md

Evolutionary Computing term project (team of ≤4, application-style).

**Title:** Neuroevolution of Autonomous Racing Agents: Generalization Across Track Geometries

## How to help

- Terse and structured. Skip preamble.
- Assume DEAP basics from class demos are known.

## What we're building

- 2D top-down racing sim in pygame.
- Car driven by a small neural net reading raycast distance sensors.
- Evolve weights with a GA (DEAP), and separately with NEAT-Python.
- Train on 3 tracks, test on a 4th unseen track, report the generalization gap.
- **Not** using Gymnasium CarRacing-v3 (image observations would force a CNN).

## Design decisions (settled — don't relitigate)

- **Physics:** arcade, not simulation. Speed, friction, turn rate scaled by speed. No drift, tire slip, or gears.
- **Sensors:** 7 raycasts at `[-90, -45, -20, 0, 20, 45, 90]` degrees relative to heading, normalized to `[0, 1]` by `MAX_RANGE`. Spatial grid so each ray only tests nearby segments.
- **Network:** 8 inputs (7 sensors + normalized speed) → 8 hidden (tanh) → 2 outputs (steer, throttle). 90 weights, flat float genome.
- **Fitness:** `checkpoints_passed * 100 + (checkpoints_passed / frames_alive) * 500`
  - Episode ends on wall contact.
  - Episode ends if no new checkpoint in ~150 frames.
  - Hard frame cap.
  - Never reward raw distance traveled.
- **Episode seed:** picks the spawn checkpoint + heading jitter (±5°). Every genome in a generation gets the same seed. `seed=None` = canonical start (checkpoint 0), for manual drive and videos.
- **Multi-track fitness:** mean of per-track fitness over the 3 training tracks.
- **Track split:** train on `oval`, `rounded_rect`, `wavy`; hold out `peanut`. `peanut` was redesigned so it is harder than the training set (tightest turn ~66 px radius vs ~75 px on `wavy`; asymmetric ends; pinched waist).
- **Generalization gap:** report in lap fractions (checkpoints passed / checkpoints per lap) alongside raw fitness, since tracks have different checkpoint counts (25/30/33/35).
- **Experiments:**
  - Baselines: random weights (matched eval budget), hand-coded heuristic.
  - 30 runs; report mean/std and convergence curves.
  - Ablations: population size, mutation sigma, selection scheme, elitism.

## Team split

1. Environment — track tool, physics, raycasts, spatial grid
2. DEAP GA
3. NEAT-Python
4. Experiments, plots, video, writeup

## Timeline

| Week | Milestone |
|------|-----------|
| W1 | Drivable car + raycast debug overlay. **Gate: nothing else starts until this works.** |
| W2 | GA completing laps; parallel eval via `multiprocessing.Pool` → `toolbox.register("map", pool.map)` |
| W3 | NEAT running |
| W4 | 30-seed sweeps + held-out track eval |
| W5 | Ablations, champion videos |
| W6 | Writeup + demo |

## Known traps

- Wrong raycast distances look like GA failure. Draw rays on screen and test against a 4-wall box before writing any EA code.
- Every fitness shape has a degenerate exploit. Expect tuning rounds.
- No `render()` or `clock.tick()` during evolution. Target: 100 generations in under 10 minutes.
- Seed episodes so all genomes in a generation face identical conditions.

## Stretch goal

If core is done by W3: evolve the tracks too, then co-evolve tracks and drivers.
