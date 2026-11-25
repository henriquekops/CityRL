import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from .city import CITY, DEFAULT_SCENARIO, SCENARIOS
from .simulator import DISTANCE_REWARD, HOLE_PENALTY, Simulation, chosen_arrivals, random_arrivals
from .traffic_light import LongestQueueLights, TrafficLightAgent
from .traffic_light_dqn import TrafficLightDQN
from .agent_router import AgentRouter

MIN_EPSILON = 0.02
EXPLORATION_DECAY_FRACTION = 0.7
RESULTS_FILE = Path("results/experiments/training.json")
SEED_STRIDE = 100_000
EVALUATION_SEED = 10_000_000
EVALUATION_EPSILON = 0.05
PRETRAIN_SEED = 20_000_000
ROUTER_AVERAGE_FRACTION = 0.3
Q_LEARNING, DQN_METHOD, LONGEST_QUEUE = "Q-learning", "DQN", "Longest queue"
METHODS = (Q_LEARNING, DQN_METHOD, LONGEST_QUEUE)


def _setting(default, group, label, minimum=None, maximum=None, step=None):
    return field(default=default, metadata=dict(group=group, label=label, range=(minimum, maximum, step)))


def _choice(default, group, label, choices):
    return field(default=default, metadata=dict(group=group, label=label, choices=choices))


@dataclass
class Hyperparameters:
    scenario: str = _choice(DEFAULT_SCENARIO, "Environment", "scenario", tuple(SCENARIOS))
    light_method: str = _choice(Q_LEARNING, "Traffic light", "method", METHODS)
    light_learning_rate: float = _setting(0.1, "Traffic light", "learning rate α", 0.01, 1, 0.05)
    light_discount: float = _setting(0.9, "Traffic light", "discount γ", 0.1, 0.999, 0.05)
    light_epsilon: float = _setting(0.3, "Traffic light", "initial ε", 0.02, 1, 0.05)
    decision_period: int = _setting(3, "Traffic light", "ticks per decision", 1, 20, 1)
    light_rich_state: bool = _setting(True, "Traffic light", "DQN: rich state")
    light_train_every: int = _setting(4, "Traffic light", "DQN: train every (transitions)", 1, 256, 1)
    fairness_weight: float = _setting(0.1, "Traffic light", "reward: longest-wait weight", 0, 2, 0.05)
    router_learning_rate: float = _setting(0.1, "Agent", "learning rate α", 0.01, 1, 0.05)
    router_discount: float = _setting(0.99, "Agent", "discount γ", 0.1, 0.999, 0.01)
    router_epsilon: float = _setting(0.05, "Agent", "initial ε", 0.02, 1, 0.05)
    router_pretrain_epochs: int = _setting(0, "Agent", "pretrain, then freeze (epochs)", 0, 5000, 50)
    dqn_learning_rate: float = _setting(0.001, "DQN", "learning rate (Adam)", 0.00001, 0.1, 0.0005)
    dqn_hidden_units: int = _setting(256, "DQN", "hidden units (2 layers)", 8, 512, 8)
    dqn_batch_size: int = _setting(128, "DQN", "batch size", 8, 1024, 8)
    dqn_memory_size: int = _setting(200000, "DQN", "replay memory", 1000, 1000000, 5000)
    dqn_target_update: int = _setting(500, "DQN", "target update (steps)", 10, 100000, 50)

    def dqn_settings(self, train_every):
        return dict(learning_rate=self.dqn_learning_rate, hidden_units=self.dqn_hidden_units,
                    batch_size=self.dqn_batch_size, memory_size=self.dqn_memory_size,
                    target_update=self.dqn_target_update, train_every=train_every)


class Trainer:
    def __init__(self, hyperparameters=None, seed=0, hole_seed=None):
        self.hyperparameters = hp = hyperparameters or Hyperparameters()
        self.seed = seed
        CITY.configure(hp.scenario, hole_seed)
        if hp.light_method == DQN_METHOD:
            self.light_agent = TrafficLightDQN(hp.light_discount, hp.dqn_settings(hp.light_train_every),
                                               hp.light_rich_state, seed, hp.fairness_weight)
        elif hp.light_method == LONGEST_QUEUE:
            self.light_agent = LongestQueueLights(hp.fairness_weight)
        else:
            self.light_agent = TrafficLightAgent(hp.light_learning_rate, hp.light_discount, seed,
                                                 hp.fairness_weight)
        self.router = AgentRouter(hp.router_learning_rate, hp.router_discount, seed)
        self.router_frozen = False
        self.history = []
        self.run_id = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @property
    def epochs_done(self):
        return len(self.history)

    def pretrain_router(self, n_agents):
        epochs = self.hyperparameters.router_pretrain_epochs
        lights, self.light_agent = self.light_agent, LongestQueueLights(self.hyperparameters.fairness_weight)
        averaged, n_averaged = None, 0
        for epoch_index in range(epochs):
            self.router.epsilon = max(MIN_EPSILON, self.hyperparameters.router_epsilon
                                      * (1 - epoch_index / (EXPLORATION_DECAY_FRACTION * epochs)))
            simulation = self._new_simulation(n_agents, seed=PRETRAIN_SEED + self.seed * SEED_STRIDE + epoch_index)
            self._run_epoch(simulation, learning=True)
            if epoch_index >= epochs * (1 - ROUTER_AVERAGE_FRACTION):
                n_averaged += 1
                averaged = (self.router.q_values.copy() if averaged is None
                            else averaged + (self.router.q_values - averaged) / n_averaged)
        if averaged is not None:
            # The frozen table is the mean of the last epochs' tables: with a constant learning rate, the noise of the
            # waits at red lights made neighbouring cells point at each other (agents going back and forth forever).
            self.router.q_values = averaged
        self.light_agent = lights
        self.router_frozen = True

    def train(self, epochs, n_agents, on_epoch=None, should_stop=lambda: False, on_tick=None, save=True):
        if self.hyperparameters.router_pretrain_epochs and not self.router_frozen:
            self.pretrain_router(n_agents)
        for epoch_index in range(epochs):
            self._set_exploration(epoch_index, epochs)
            simulation = self._new_simulation(n_agents, seed=self.seed * SEED_STRIDE + self.epochs_done)
            seconds = self._run_epoch(simulation, learning=True, should_stop=should_stop, on_tick=on_tick)
            if should_stop():
                break
            self.history.append(self._epoch_record(simulation, n_agents, seconds))
            if on_epoch:
                on_epoch(self.history[-1])
        if self.history and save:
            self.save_results()

    def _set_exploration(self, epoch_index, epochs):
        decay = 1 - epoch_index / (EXPLORATION_DECAY_FRACTION * epochs)
        hp = self.hyperparameters
        self.light_agent.epsilon = max(MIN_EPSILON, hp.light_epsilon * decay)
        self.router.epsilon = MIN_EPSILON if self.router_frozen else max(MIN_EPSILON, hp.router_epsilon * decay)

    def _new_simulation(self, n_agents, seed):
        return Simulation(random_arrivals(n_agents, seed), self.hyperparameters.decision_period)

    def _run_epoch(self, simulation, learning, should_stop=lambda: False, on_tick=None, explore=None):
        explore = learning if explore is None else explore
        self.router.learning = learning and not self.router_frozen
        self.router.exploring = explore
        self.light_agent.start_episode()
        seconds = 0.0
        while not simulation.is_done() and not should_stop():
            started = time.perf_counter()
            self.step(simulation, learning, explore)
            seconds += time.perf_counter() - started
            if on_tick:
                on_tick(simulation, self.epochs_done + 1)
        return seconds

    def step(self, simulation, learning=False, explore=None):
        if simulation.time % simulation.decision_period == 0:
            self.light_agent.decide(simulation, learning, explore)
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
            light_return=round(simulation.light_return, 2), router_return=round(simulation.router_return, 2),
            light_loss=self._mean_loss(self.light_agent),
            light_epsilon=round(self.light_agent.epsilon, 3), router_epsilon=round(self.router.epsilon, 3))

    @staticmethod
    def _mean_loss(learner):
        loss = learner.pop_mean_loss() if hasattr(learner, "pop_mean_loss") else None
        return None if loss is None else round(loss, 5)

    def evaluate(self, episodes, n_agents):
        saved = self.light_agent.epsilon, self.router.epsilon
        self.light_agent.epsilon, self.router.epsilon = EVALUATION_EPSILON, MIN_EPSILON
        records = []
        for episode in range(episodes):
            simulation = self._new_simulation(n_agents, seed=EVALUATION_SEED + episode)
            self._run_epoch(simulation, learning=False, explore=True)
            records.append(dict(
                arrived=len(simulation.trips), agents_created=len(simulation.arrivals),
                hole_hits=simulation.hole_hits, mean_trip_ticks=simulation.mean_trip_ticks(),
                mean_wait_ticks=simulation.mean_wait_ticks(), makespan_ticks=simulation.time,
                light_return=simulation.light_return, router_return=simulation.router_return))
        self.light_agent.epsilon, self.router.epsilon = saved
        return {key: round(sum(record[key] for record in records) / episodes, 3) for key in records[0]}

    def simulation_for_agents(self, agents, max_agents_on_map=None):
        self.router.learning = self.router.exploring = False
        self.light_agent.start_episode()
        return Simulation(chosen_arrivals(agents), self.hyperparameters.decision_period, max_agents_on_map)

    def save_results(self, path=RESULTS_FILE):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        other_runs = [run for run in self._load_runs(path) if run["id"] != self.run_id]
        destinations = [dict(name=name, house=list(house))
                        for name, house in zip(CITY.destination_names, CITY.destination_houses)]
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
        last_light = [record["light_return"] for record in self.history[-10:]]
        last_router = [record["router_return"] for record in self.history[-10:]]
        return dict(
            id=self.run_id, seed=self.seed, hyperparameters=asdict(self.hyperparameters),
            environment=dict(scenario=CITY.scenario.name, crossings=CITY.n_intersections,
                             holes=[list(hole) for hole in CITY.holes], hole_penalty=HOLE_PENALTY,
                             distance_reward=DISTANCE_REWARD),
            epochs_trained=self.epochs_done,
            total_ticks_trained=self.history[-1]["cumulative_ticks"] if self.history else 0,
            total_seconds_trained=round(sum(record["seconds"] for record in self.history), 1),
            final_mean_trip_ticks=round(sum(last_trips) / len(last_trips), 2) if last_trips else None,
            final_light_return=round(sum(last_light) / len(last_light), 2) if last_light else None,
            final_router_return=round(sum(last_router) / len(last_router), 2) if last_router else None,
            epochs=self.history)
