# CityRL

A traffic gridworld.

Streets forming blocks, a traffic light at each crossing, destination houses and holes. The
map starts with 6 streets, 9 crossings, 4 houses and one hole, and larger scenarios scale
it up to 64 crossings. Agents enter at the border, walk to a destination and leave
the simulation when they reach its door.

## How to run

```bash
uv run python main.py
```

## Experiments

Every scenario, every method, same seeds (same traffic) and same holes. The router is pretrained once per scenario
and seed and reused by all methods:

```bash
uv run python -m cityrl.experiments --epochs 150 --seeds 3
uv run python -m cityrl.plots   # results/figures: one figure per scenario, scale.png and evaluation.csv
```

`--set name=value` overrides a hyperparameter (e.g. `--set dqn_hidden_units=128 fairness_weight=0`). The
evaluation runs the learned policy, without learning and with ε = 0.05, on 10-20 episodes never used in training.


## Scenarios

| Scenario                  | Map   | Crossings | Destinations | Holes | Agents/episode |
|---------------------------|-------|-----------|--------------|-------|----------------|
| `T1: 9 crossings (11x11)` | 11×11 | 9         | 4            | 1     | 300            |
| `16 crossings (16x16)`    | 16×16 | 16        | 6            | 2     | 900            |
| `36 crossings (26x26)`    | 26×26 | 36        | 8            | 4     | 2200           |
| `64 crossings (36x36)`    | 36×36 | 64        | 10           | 6     | 4000           |

The scenario is chosen in the interface (Environment group) or with `--scenarios`. Holes are placed at most one
per street segment, so every door stays reachable.
