"""Joint training of the traffic lights and the agent router, epoch by epoch, with results saved as JSON."""
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from .city import CITY, DESTINATION_HOUSES, DESTINATION_NAMES
from .simulator import DISTANCE_REWARD, HOLE_PENALTY, Simulation, chosen_arrivals, random_arrivals
from .traffic_light import TrafficLightAgent
from .agent_router import AgentRouter

MIN_EPSILON = 0.02                  # exploration never drops below this
EXPLORATION_DECAY_FRACTION = 0.7    # epsilon decays linearly until this fraction of the epochs
RESULTS_FILE = Path("results/training.json")


def _setting(default, group, label, minimum=None, maximum=None, step=None):
    """A hyperparameter with the information the interface needs to show it (group, label and allowed range)."""
    return field(default=default, metadata=dict(group=group, label=label, range=(minimum, maximum, step)))


@dataclass
class Hyperparameters:
    """Every value shown in the interface. The defaults come from an earlier one-factor-at-a-time search, made
    with a different traffic-light state, and were not re-tuned for the current one."""
    light_learning_rate: float = _setting(0.1, "Traffic light", "learning rate α", 0.01, 1, 0.05)     # weight of each correction
    light_discount: float = _setting(0.9, "Traffic light", "discount γ", 0.1, 0.999, 0.05)           # weight of the future
    light_epsilon: float = _setting(0.3, "Traffic light", "initial ε", 0.02, 1, 0.05)                # exploration at the start
    decision_period: int = _setting(3, "Traffic light", "ticks per decision", 1, 20, 1)
    router_learning_rate: float = _setting(0.1, "Agent", "learning rate α", 0.01, 1, 0.05)
    router_discount: float = _setting(0.99, "Agent", "discount γ", 0.1, 0.999, 0.01)              # per tick
    router_epsilon: float = _setting(0.05, "Agent", "initial ε", 0.02, 1, 0.05)
    backward_propagation: bool = _setting(True, "Agent", "backward propagation")                  # correct the whole trip
    propagation_threshold: float = _setting(0.05, "Agent", "threshold ω", 0, 5, 0.05)             # stop below this size


class Trainer:
    def __init__(self, hyperparameters=None):
        self.hyperparameters = hp = hyperparameters or Hyperparameters()
        self.light_agent = TrafficLightAgent(hp.light_learning_rate, hp.light_discount)
        self.router = AgentRouter(hp.router_learning_rate, hp.router_discount,
                                    hp.propagation_threshold if hp.backward_propagation else None)
        self.history = []                                           # one record per trained epoch
        self.run_id = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @property
    def epochs_done(self):
        return len(self.history)

    # ---------- training ----------
    def train(self, epochs, n_agents, on_epoch=None, should_stop=lambda: False):
        """Train for `epochs` epochs of `n_agents` agents each, then save the run.
        `on_epoch(record)` is called after each epoch. When `should_stop()` turns true, training stops right away:
        the unfinished epoch is discarded (it is not recorded) and the finished ones are saved."""
        for epoch_index in range(epochs):
            self._set_exploration(epoch_index, epochs)
            simulation = self._new_simulation(n_agents, seed=self.epochs_done)    # same seed per epoch in every run
            started = time.perf_counter()
            self._run_epoch(simulation, learning=True, should_stop=should_stop)
            if should_stop():
                break
            seconds = time.perf_counter() - started
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

    def _run_epoch(self, simulation, learning, should_stop=lambda: False):
        self.router.learning = learning
        self.light_agent.start_episode()
        while not simulation.is_done() and not should_stop():
            self.step(simulation, learning)

    def step(self, simulation, learning=False):
        """One tick: the traffic lights decide when it is time, then the simulator moves the agents."""
        if simulation.time % simulation.decision_period == 0:
            self.light_agent.decide(simulation, learning)
        simulation.tick(self.router)

    def _epoch_record(self, simulation, n_agents, seconds):
        """`agents_created` can be a bit below `agents_requested`: entries that would start on their own door are skipped."""
        return dict(
            epoch=self.epochs_done + 1, agents_requested=n_agents, agents_created=len(simulation.arrivals),
            arrived=len(simulation.trips), hole_hits=simulation.hole_hits,
            mean_trip_ticks=round(simulation.mean_trip_ticks(), 2),
            mean_wait_ticks=round(simulation.mean_wait_ticks(), 2),
            makespan_ticks=simulation.time,                          # ticks until the last agent left
            cumulative_ticks=sum(r["makespan_ticks"] for r in self.history) + simulation.time,
            seconds=round(seconds, 2),                               # wall-clock time of the epoch
            light_epsilon=round(self.light_agent.epsilon, 3), router_epsilon=round(self.router.epsilon, 3))

    # ---------- visualization ----------
    def simulation_for_agents(self, agents):
        """A simulation with only the given agents [(cell, destination)], run with the learned policy."""
        self.router.learning = False
        self.light_agent.start_episode()
        return Simulation(chosen_arrivals(agents), self.hyperparameters.decision_period)

    # ---------- results ----------
    def save_results(self, path=RESULTS_FILE):
        """Add (or update) this run in the results file, next to the previous runs, so runs can be compared."""
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
