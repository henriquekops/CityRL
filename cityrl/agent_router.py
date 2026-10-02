"""Agent router: tabular Q-learning on grid cells (the navigation setting of Hafez and Loo, 2015)."""
import numpy as np

from .city import CITY, DESTINATION_HOUSES, GRID_SIZE


class AgentRouter:
    """One Q-table per destination, shared by all agents headed there. State = cell; action = direction.

    After each move, the Q-value of that move is corrected towards  reward + discount * (best value of the next cell)."""

    def __init__(self, learning_rate, discount, seed=0):
        self.learning_rate, self.discount = learning_rate, discount
        self.q_values = np.zeros((len(DESTINATION_HOUSES), GRID_SIZE * GRID_SIZE, 4))     # [destination, cell, direction]
        self.rng = np.random.default_rng(seed)
        self.epsilon = 0.0              # probability of a random move while learning
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
        """Called after every tick with the reward of the move the agent chose; the agent is now on its new cell."""
        if not self.learning:
            return
        cell, direction = agent.last_decision
        future = 0.0 if is_final else self.discount * self._best_value(agent.destination, agent.cell)
        old_value = self.q_values[agent.destination, cell, direction]
        self.q_values[agent.destination, cell, direction] += self.learning_rate * (reward + future - old_value)

    def _best_value(self, destination, cell):
        """Value of a cell: the best Q-value among its valid moves."""
        return np.max(np.where(CITY.valid_moves[cell], self.q_values[destination, cell], -np.inf))

    def _best_move(self, values, valid_moves):
        values = np.where(valid_moves, values, -np.inf)
        best_moves = np.flatnonzero(values >= values.max() - 1e-12)
        return int(self.rng.choice(best_moves))
