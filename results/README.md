# Results

Saved runs behind the numbers in the report. Each `seedN/` folder holds one run:
`config.json`, `logbook.csv` (per-generation fitness), `champion.json` (+ `champion.pkl`
and `neat_config.ini` for NEAT) and `summary.json` (champion scored on all 8 tracks,
10 test seeds each).

| Folder | What | Seeds | Code |
|---|---|---|---|
| `ga/default` | DEAP GA, fixed sigma 0.3 | 0-29 | 5049ddd |
| `neat/default` | NEAT-Python | 0-29 | 5049ddd |
| `ga/sigma_schedule` | GA, sigma shrinks 0.3 -> 0.05 over the run | 0-4 | `--sigma-control schedule` (c74ed3e) |
| `ga/sigma_self_adaptive` | GA, each genome carries its own sigma | 0-4 | `--sigma-control self_adaptive` (c74ed3e) |
| `eval/` | Comparison tables and held-out scores (below) | | |

All runs: pop 100 x 100 generations, trained on the 7 `TRAIN_TRACKS`, 4 worker processes.
Minutes per run are rough: some runs were on battery.

## GA vs NEAT (30 seeds each)

"Clean" = the champion laps the track with 0 crashes over the 10 test seeds.

| | Peanut laps | Peanut clean | hairpin_brake laps | hairpin_brake clean | Training crash rate |
|---|---|---|---|---|---|
| GA | 2.07 ± 1.02 | 16/30 | 0.41 ± 0.41 | 3/30 | 18.2% |
| NEAT | 2.82 ± 0.47 | 28/30 | 1.42 ± 0.51 | 23/30 | 0.5% |
| Hand-coded | 2.85 (0 crashes) | | 1.63 (0 crashes) | | |

- NEAT is better on both unseen tracks: p < 0.001 for laps (permutation test) and clean seeds (Fisher) on each.
- GA champions mostly learned "always top speed" and hit the wall at tight turns after long straights.
  NEAT champions slow down; when they fail on hairpin_brake they stall in the hairpin rather than crash.
- GA is all-or-nothing: most seeds either lap peanut cleanly or crash in all 10 episodes.

Files: `eval/ga_vs_neat.txt`, `eval/ga_held_out.txt`, `eval/neat_held_out.txt` (per-seed scores on both held-out tracks).

## First 5 seeds vs the other 25

No significant difference for either method, so seeds 0-4 were a fair sample;
the extra seeds made GA vs NEAT significant (p = 0.39 at 5 seeds, < 0.001 at 30).

| | Seeds 0-4 peanut laps / clean | Seeds 5-29 peanut laps / clean | p (laps) |
|---|---|---|---|
| GA | 2.16 / 3 of 5 | 2.05 / 13 of 25 | 0.70 |
| NEAT | 2.57 / 4 of 5 | 2.87 / 24 of 25 | 0.11 |

Files: `eval/ga_first5_vs_rest.txt`, `eval/neat_first5_vs_rest.txt`.

## GA parameter control (5 seeds each)

| Sigma control | Peanut laps | Peanut clean | Training crash rate |
|---|---|---|---|
| Fixed (default) | 2.16 ± 1.01 | 3/5 | 21.7% |
| Shrinking schedule | 1.71 ± 1.20 | 2/5 | 31.7% |
| Self-adaptive | 2.55 ± 1.04 | 4/5 | 15.4% |

No significant difference (p >= 0.48). Self-adaptive sigma stayed between about 0.25 and 0.35,
so it barely adapted. File: `eval/sigma_control.txt`.

## Second held-out track: hairpin_brake

Added in `simple_tracks.py` (never trained on). Two long straights each end in a 180 deg,
85 px hairpin: a braking test. The 85 px radius was chosen after trying 65 and 75 px so NEAT
clears it about as often as peanut; GA failed at every radius, so the GA vs NEAT result does not
depend on that choice. Note for the writeup: the difficulty was tuned after seeing champion scores.

## Reproduce

```
python -m evo.ga_deap --seed 0 --processes 4 --out results/ga/default
python -m evo.neat_run --seed 0 --processes 4 --out results/neat/default
python -m evo.ga_deap --seed 0 --processes 4 --sigma-control self_adaptive --out results/ga/sigma_self_adaptive
python -m experiments.compare "GA=results/ga/default" "NEAT=results/neat/default"
python -m experiments.evaluate results/ga/default/seed*/champion.json --tracks peanut,hairpin_brake
python -m evo.replay results/neat/default/seed0/champion.json --track hairpin_brake
```
