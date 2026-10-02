# CityRL

A small traffic gridworld: 6 streets (3 horizontal, 3 vertical) forming 4 blocks, a traffic light at each of the
9 crossings, 4 destination houses (A-D) and one hole. Agents enter at the border, walk to a destination and leave
the simulation when they reach its door.

**Rules.** Agents do not interact with each other (they may share a cell). Each tick an agent chooses north, east,
south or west:
- moving into the hole is blocked: the agent stays and gets a negative reward (-10). The hole is drawn at random on
  any street cell (never on an intersection or a destination door) each time the program starts; set `HOLE_SEED` in
  `city.py` to fix it, which makes runs from different sessions comparable;
- moving into an intersection on red makes the agent wait;
- otherwise the move happens.

Every tick costs -1. On top of that, each cell closer to the destination pays +0.5 and each cell farther costs -0.5
(*reward shaping* on the straight-line distance, which ignores blocks and the hole), so agents learn short routes that
avoid the hole faster.

**Learning** (all tabular Q-learning, no model of the world):
- *Traffic lights* (Jin et al., 2011): state = agents on the north/south arms, agents on the east/west arms and the
  current signal; action = which axis gets green, decided every N ticks. An intersection with no agents on its arms
  closes (all red) by rule.
- *Agents* (inspired by Hafez and Loo, 2015): state = cell, action = direction, one Q-table per destination shared by
  all agents headed there.

```bash
uv run python main.py
```

Everything is controlled in the interface: agents per epoch, epochs, hyperparameters, training and visualization of up
to 10 agents (click on a street cell to add or remove one; intersections, the hole and houses are not allowed).
The traffic lights are always drawn.

## Comparing hyperparameters

Each run (Train, then New run, change values, Train) is saved to `results/training.json` under `runs`, with its
hyperparameters, the environment (hole, hole penalty, distance reward) and one record per epoch (agents requested and
created, arrivals, hole hits, mean trip and wait in ticks, epoch length, cumulative ticks, wall-clock seconds, epsilons), plus `final_mean_trip_ticks` (mean of the last 10
epochs). Epoch *i* always uses the same scenario seed, so runs with the same agents per epoch are comparable. Time is
measured in *ticks* (one tick = every agent tries to move one cell).

## Code

| File | Role |
|---|---|
| `city.py` | the map: street cells, intersections, hole, destination doors |
| `simulator.py` | the rules, one `tick()` at a time (does not learn) |
| `traffic_light.py` | `TrafficLightAgent` |
| `agent_router.py` | `AgentRouter` (Q-learning) |
| `trainer.py` | `Hyperparameters`, epoch loop, JSON results |
| `ui.py` | tkinter interface (`MapView` draws, `App` controls) |
