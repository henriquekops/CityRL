import numpy as np

from .city import BLOCK_SIZE
from .dqn import DQN
from .traffic_light import MAX_COUNTED_AGENTS, LightController

N_SIGNALS = 2
MAX_COUNTED_TICKS = 30


class TrafficLightDQN(LightController):

    def __init__(self, discount, dqn_settings, rich_state=True, seed=0, fairness_weight=0.0):
        self.rich_state = rich_state
        n_inputs = 4 + 3 + 1 + 1 + 4 if rich_state else 2 + 3
        self.dqn = DQN(n_inputs, N_SIGNALS, discount, seed=seed, **dqn_settings)
        super().__init__(fairness_weight)
        self.last_change = {}

    @property
    def epsilon(self):
        return self.dqn.epsilon

    @epsilon.setter
    def epsilon(self, value):
        self.dqn.epsilon = value

    def start_episode(self):
        super().start_episode()
        self.last_change = {}

    def _decide(self, sim, intersection, learning, explore, cost, north_south, east_west, longest_wait):
        state = self._state(sim, intersection, north_south, east_west, longest_wait)
        if learning and intersection in self.previous:
            previous_state, previous_action = self.previous[intersection]
            self.dqn.remember(previous_state, previous_action, -cost, state, done=False)
        action = int(self.dqn.act(state, explore=explore)[0])
        self.previous[intersection] = (state, action)
        if action != sim.signal[intersection]:
            self.last_change[intersection] = sim.time
        return action

    def _state(self, sim, intersection, north_south, east_west, longest_wait):
        signal = np.zeros(3, np.float32)
        signal[sim.signal[intersection] + 1] = 1.0
        if not self.rich_state:
            capped = np.array([min(north_south, MAX_COUNTED_AGENTS), min(east_west, MAX_COUNTED_AGENTS)])
            return np.concatenate([capped / MAX_COUNTED_AGENTS, signal]).astype(np.float32)
        arms = [self._queue(sim, arm) / BLOCK_SIZE for arm in sim.city.arms[intersection]]
        since_change = min(sim.time - self.last_change.get(intersection, 0), MAX_COUNTED_TICKS) / MAX_COUNTED_TICKS
        neighbours = [sum(self._queue(sim, arm) for arm in sim.city.arms[other]) / (4 * BLOCK_SIZE) if other >= 0
                      else 0.0 for other in sim.city.neighbor_intersection[intersection]]
        waited = min(longest_wait, MAX_COUNTED_TICKS) / MAX_COUNTED_TICKS
        return np.concatenate([arms, signal, [since_change, waited], neighbours]).astype(np.float32)

    @staticmethod
    def _queue(sim, arm):
        return sum(sim.agents_per_cell[cell] for cell in arm)

    def pop_mean_loss(self):
        return self.dqn.pop_mean_loss()
