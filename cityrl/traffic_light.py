from collections import defaultdict

import numpy as np

from .simulator import CLOSED

MAX_COUNTED_AGENTS = 4


class TrafficLightAgent:

    def __init__(self, learning_rate, discount, seed=0):
        self.learning_rate, self.discount = learning_rate, discount
        self.q_table = defaultdict(lambda: np.zeros(2))
        self.rng = np.random.default_rng(seed)
        self.epsilon = 0.0
        self.previous = {}

    def start_episode(self):
        self.previous = {}

    def decide(self, sim, learning):
        for intersection in range(sim.city.n_intersections):
            north_south, east_west = sim.waiting_by_axis(intersection)
            if north_south + east_west == 0:
                sim.set_signal(intersection, CLOSED)
                self.previous.pop(intersection, None)
            else:
                self._decide_with_q_table(sim, intersection, learning, north_south, east_west)
            sim.waiting_since_decision[intersection] = 0.0

    def _decide_with_q_table(self, sim, intersection, learning, north_south, east_west):
        signal = sim.signal[intersection]
        state = (min(north_south, MAX_COUNTED_AGENTS), min(east_west, MAX_COUNTED_AGENTS), signal)
        if learning and intersection in self.previous:
            queue_cost = sim.waiting_since_decision[intersection] / sim.decision_period
            self._update_q(*self.previous[intersection], queue_cost, state)
        action = self._choose_action(state, learning, signal, north_south > east_west)
        self.previous[intersection] = (state, action)
        sim.set_signal(intersection, action)

    def _update_q(self, state, action, queue_cost, next_state):
        target = -queue_cost + self.discount * self.q_table[next_state].max()
        self.q_table[state][action] += self.learning_rate * (target - self.q_table[state][action])

    def _choose_action(self, state, learning, current_signal, north_south_busier):
        values = self.q_table[state]
        if learning and self.rng.random() < self.epsilon:
            return int(self.rng.integers(2))
        if values[0] == values[1]:
            return current_signal if current_signal != CLOSED else int(north_south_busier)
        return int(np.argmax(values))
