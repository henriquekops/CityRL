from collections import defaultdict

import numpy as np

from .simulator import CLOSED, EAST_WEST, NORTH_SOUTH

MAX_COUNTED_AGENTS = 4


class LightController:

    def __init__(self, fairness_weight=0.0):
        self.fairness_weight = fairness_weight
        self.epsilon = 0.0
        self.previous = {}

    def start_episode(self):
        self.previous = {}

    def decide(self, sim, learning, explore=None):
        explore = learning if explore is None else explore
        longest_waits = sim.longest_waits()
        for intersection in range(sim.city.n_intersections):
            cost = sim.decision_cost(intersection, longest_waits[intersection], self.fairness_weight)
            north_south, east_west = sim.waiting_by_axis(intersection)
            if north_south + east_west == 0:
                sim.set_signal(intersection, CLOSED)
                self.previous.pop(intersection, None)
            else:
                signal = self._decide(sim, intersection, learning, explore, cost, north_south, east_west,
                                      longest_waits[intersection])
                sim.set_signal(intersection, signal)
            sim.waiting_since_decision[intersection] = 0.0

    def _decide(self, sim, intersection, learning, explore, cost, north_south, east_west, longest_wait):
        raise NotImplementedError


class LongestQueueLights(LightController):

    def _decide(self, sim, intersection, learning, explore, cost, north_south, east_west, longest_wait):
        if north_south == east_west and sim.signal[intersection] != CLOSED:
            return sim.signal[intersection]
        return NORTH_SOUTH if north_south > east_west else EAST_WEST


class TrafficLightAgent(LightController):

    def __init__(self, learning_rate, discount, seed=0, fairness_weight=0.0):
        super().__init__(fairness_weight)
        self.learning_rate, self.discount = learning_rate, discount
        self.q_table = defaultdict(lambda: np.zeros(2))
        self.rng = np.random.default_rng(seed)

    def _decide(self, sim, intersection, learning, explore, cost, north_south, east_west, longest_wait):
        signal = sim.signal[intersection]
        state = (min(north_south, MAX_COUNTED_AGENTS), min(east_west, MAX_COUNTED_AGENTS), signal)
        if learning and intersection in self.previous:
            self._update_q(*self.previous[intersection], cost, state)
        action = self._choose_action(state, explore, signal, north_south > east_west)
        self.previous[intersection] = (state, action)
        return action

    def _update_q(self, state, action, cost, next_state):
        target = -cost + self.discount * self.q_table[next_state].max()
        self.q_table[state][action] += self.learning_rate * (target - self.q_table[state][action])

    def _choose_action(self, state, explore, current_signal, north_south_busier):
        values = self.q_table[state]
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(2))
        if values[0] == values[1]:
            return current_signal if current_signal != CLOSED else int(north_south_busier)
        return int(np.argmax(values))
