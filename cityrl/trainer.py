import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from .city import CITY, DESTINATION_HOUSES, DESTINATION_NAMES
from .simulator import DISTANCE_REWARD, HOLE_PENALTY, Simulation, chosen_arrivals, random_arrivals
from .traffic_light import TrafficLightAgent
from .agent_router import AgentRouter

MIN_EPSILON = 0.02
EXPLORATION_DECAY_FRACTION = 0.7
RESULTS_FILE = Path("results/training.json")


def _setting(default, group, label, minimum=None, maximum=None, step=None):
    return field(default=default, metadata=dict(group=group, label=label, range=(minimum, maximum, step)))


@dataclass
class Hyperparameters:
    light_learning_rate: float = _setting(0.1, "Traffic light", "learning rate α", 0.01, 1, 0.05)
    light_discount: float = _setting(0.9, "Traffic light", "discount γ", 0.1, 0.999, 0.05)
    light_epsilon: float = _setting(0.3, "Traffic light", "initial ε", 0.02, 1, 0.05)
    decision_period: int = _setting(3, "Traffic light", "ticks per decision", 1, 20, 1)
    router_learning_rate: float = _setting(0.1, "Agent", "learning rate α", 0.01, 1, 0.05)
    router_discount: float = _setting(0.99, "Agent", "discount γ", 0.1, 0.999, 0.01)
    router_epsilon: float = _setting(0.05, "Agent", "initial ε", 0.02, 1, 0.05)


class Trainer:
    def __init__(self, hyperparameters=None):
        self.hyperparameters = hp = hyperparameters or Hyperparameters()
        self.light_agent = TrafficLightAgent(hp.light_learning_rate, hp.light_discount)
        self.router = AgentRouter(hp.router_learning_rate, hp.router_discount)
        self.history = []
        self.run_id = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @property
    def epochs_done(self):
        return len(self.history)

    def train(self, epochs, n_agents, on_epoch=None, should_stop=lambda: False, on_tick=None):
        for epoch_index in range(epochs):
            self._set_exploration(epoch_index, epochs)
            simulation = self._new_simulation(n_agents, seed=self.epochs_done)
            seconds = self._run_epoch(simulation, learning=True, should_stop=should_stop, on_tick=on_tick)
            if should_stop():
                break
            self.history.append(self._epoch_record(simulation, n_agents, seconds))
            if on_epoch:
                on_epoch(self.history[-1])
        if self.history:
            self.save_results()

    def _set_exploration(self, epoch_index, epochs):
        decay = 1 - epoch_index / (EXPLORATION_DECAY_FRACTION * epochs)
        hp = self.hyperparameters
        self.light_agent.epsilon = max(MIN_EPSILON, hp.light_epsilon * decay)
        self.router.epsilon = max(MIN_EPSILON, hp.router_epsilon * decay)

    def _new_simulation(self, n_agents, seed):
        return Simulation(random_arrivals(n_agents, seed), self.hyperparameters.decision_period)

    def _run_epoch(self, simulation, learning, should_stop=lambda: False, on_tick=None):
        self.router.learning = learning
        self.light_agent.start_episode()
        seconds = 0.0
        while not simulation.is_done() and not should_stop():
            started = time.perf_counter()
            self.step(simulation, learning)
            seconds += time.perf_counter() - started
            if on_tick:
                on_tick(simulation, self.epochs_done + 1)
        return seconds

    def step(self, simulation, learning=False):
        if simulation.time % simulation.decision_period == 0:
            self.light_agent.decide(simulation, learning)
        simulation.tick(self.router)

    def _epoch_record(self, simulation, n_agents, seconds):
        return dict(
            epoch=self.epochs_done + 1, agents_requested=n_agents, agents_created=len(simulation.arrivals),
            arrived=len(simulation.trips), hole_hits=simulation.hole_hits,
            mean_trip_ticks=round(simulation.mean_trip_ticks(), 2),
            mean_wait_ticks=round(simulation.mean_wait_ticks(), 2),
            makespan_ticks=simulation.time,
            cumulative_ticks=sum(r["makespan_ticks"] for r in self.history) + simulation.time,
            seconds=round(seconds, 2),
            light_epsilon=round(self.light_agent.epsilon, 3), router_epsilon=round(self.router.epsilon, 3))

    def simulation_for_agents(self, agents):
        self.router.learning = False
        self.light_agent.start_episode()
        return Simulation(chosen_arrivals(agents), self.hyperparameters.decision_period)

    def save_results(self, path=RESULTS_FILE):
        path = Path(path)
        path.parent.mkdir(exist_ok=True)
        other_runs = [run for run in self._load_runs(path) if run["id"] != self.run_id]
        destinations = [dict(name=name, house=list(house)) for name, house in zip(DESTINATION_NAMES, DESTINATION_HOUSES)]
        path.write_text(json.dumps(dict(destinations=destinations, runs=other_runs + [self._run_summary()]),
                                   indent=1, ensure_ascii=False))

    @staticmethod
    def _load_runs(path):
        try:
            return json.loads(path.read_text())["runs"]
        except (OSError, ValueError, KeyError):
            return []

    def _run_summary(self):
        last_trips = [record["mean_trip_ticks"] for record in self.history[-10:]]
        return dict(
            id=self.run_id, hyperparameters=asdict(self.hyperparameters),
            environment=dict(hole=list(CITY.hole), hole_penalty=HOLE_PENALTY, distance_reward=DISTANCE_REWARD),
            epochs_trained=self.epochs_done,
            total_ticks_trained=self.history[-1]["cumulative_ticks"] if self.history else 0,
            total_seconds_trained=round(sum(record["seconds"] for record in self.history), 1),
            final_mean_trip_ticks=round(sum(last_trips) / len(last_trips), 2) if last_trips else None,
            epochs=self.history)
