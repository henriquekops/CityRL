"""Agent router: tabular Q-learning on grid cells, with backward propagation along the trip
(inspired by Hafez and Loo, 2015)."""
from typing import NamedTuple

import numpy as np

from .city import CITY, DESTINATION_HOUSES, GRID_SIZE


class Transition(NamedTuple):
    """What one move led to."""
    cell: int                       # where the agent was
    direction: int                  # the move it chose
    reward: float
    next_cell: int                  # where it ended up (the same cell if it could not move)
    is_final: bool                  # it reached the destination


class AgentRouter:
    """One Q-table per destination, shared by all agents headed there. State = cell; action = direction.

    After each move, the Q-value of that move is corrected towards  reward + discount * (best value of the next cell).
    The correction is then propagated backwards through the moves already made in the same trip while it stays above
    `propagation_threshold` (None disables it, giving plain Q-learning)."""

    def __init__(self, learning_rate, discount, propagation_threshold, seed=0):
        self.learning_rate, self.discount, self.propagation_threshold = learning_rate, discount, propagation_threshold
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
        """Called after every tick with the reward of the move the agent chose and where it is now."""
        if not self.learning:
            return
        cell, direction = agent.last_decision
        agent.transitions.append(Transition(cell, direction, reward, agent.cell, is_final))
        for steps_back, transition in enumerate(reversed(agent.transitions)):
            error = self._target(agent.destination, transition) - self._q(agent.destination, transition)
            if steps_back > 0 and (self.propagation_threshold is None or abs(error) < self.propagation_threshold):
                break
            self.q_values[agent.destination, transition.cell, transition.direction] += self.learning_rate * error

    def _q(self, destination, transition):
        return self.q_values[destination, transition.cell, transition.direction]

    def _target(self, destination, transition):
        """Reward plus the discounted value of the best move from the next cell (0 after arriving)."""
        if transition.is_final:
            return transition.reward
        next_values = self.q_values[destination, transition.next_cell]
        best_next = np.max(np.where(CITY.valid_moves[transition.next_cell], next_values, -np.inf))
        return transition.reward + self.discount * best_next

    def _best_move(self, values, valid_moves):
        values = np.where(valid_moves, values, -np.inf)
        best_moves = np.flatnonzero(values >= values.max() - 1e-12)
        return int(self.rng.choice(best_moves))
