import numpy as np

from .city import CITY, DESTINATION_HOUSES, GRID_SIZE


class AgentRouter:

    def __init__(self, learning_rate, discount, seed=0):
        self.learning_rate, self.discount = learning_rate, discount
        self.q_values = np.zeros((len(DESTINATION_HOUSES), GRID_SIZE * GRID_SIZE, 4))
        self.rng = np.random.default_rng(seed)
        self.epsilon = 0.0
        self.learning = False

    def choose_move(self, agent):
        valid_moves = CITY.valid_moves[agent.cell]
        if self.learning and self.rng.random() < self.epsilon:
            direction = int(self.rng.choice(np.flatnonzero(valid_moves)))
        else:
            direction = self._best_move(self.q_values[agent.destination, agent.cell], valid_moves)
        agent.last_decision = (agent.cell, direction)
        return direction

    def learn(self, agent, reward, is_final):
        if not self.learning:
            return
        cell, direction = agent.last_decision
        future = 0.0 if is_final else self.discount * self._best_value(agent.destination, agent.cell)
        old_value = self.q_values[agent.destination, cell, direction]
        self.q_values[agent.destination, cell, direction] += self.learning_rate * (reward + future - old_value)

    def _best_value(self, destination, cell):
        return np.max(np.where(CITY.valid_moves[cell], self.q_values[destination, cell], -np.inf))

    def _best_move(self, values, valid_moves):
        values = np.where(valid_moves, values, -np.inf)
        best_moves = np.flatnonzero(values >= values.max() - 1e-12)
        return int(self.rng.choice(best_moves))
