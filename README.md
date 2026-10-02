# CityRL

A small traffic gridworld: 6 streets (3 horizontal, 3 vertical) forming 4 blocks, a traffic light at each of the
9 crossings, 4 destination houses (A-D) and one hole. Agents enter at the border, walk to a destination and leave
the simulation when they reach its door.

## How to run

```bash
uv run python main.py
```

## Code

| File               | Role                                                          |
|--------------------|---------------------------------------------------------------|
| `city.py`          | the map. street cells, intersections, hole, destination doors |
| `simulator.py`     | the rules, one `tick()` at a time (does not learn)            |
| `traffic_light.py` | `TrafficLightAgent` (Q-learning)                              |
| `agent_router.py`  | `AgentRouter` (Q-learning)                                    |
| `trainer.py`       | `Hyperparameters`, epoch loop, JSON results                   |
| `ui.py`            | tkinter interface (`MapView` draws, `App` controls)           |
