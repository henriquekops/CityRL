from collections import Counter
from dataclasses import dataclass

import numpy as np

from .city import CITY, EAST, NORTH, SOUTH, WEST

DEFAULT_DECISION_PERIOD = 3
STEP_REWARD = -1
HOLE_PENALTY = 10
DISTANCE_REWARD = 0.5

CLOSED, EAST_WEST, NORTH_SOUTH = -1, 0, 1


@dataclass
class Arrival:
    tick: int
    destination: int
    cell: int


def random_arrivals(n_agents, seed):
    rng = np.random.default_rng(seed)
    arrivals = []
    for tick in sorted(rng.integers(0, CITY.scenario.spawn_window_ticks, n_agents)):
        destination = int(rng.integers(CITY.n_destinations))
        cell = int(rng.choice(CITY.spawn_cells))
        if cell != CITY.door_cell[destination]:
            arrivals.append(Arrival(int(tick), destination, cell))
    return arrivals


def chosen_arrivals(agents):
    return [Arrival(0, destination, cell) for cell, destination in agents]


def scattered_agents(n_agents, seed=None):
    rng = np.random.default_rng(seed)
    cells = rng.permutation(CITY.agent_start_cells())[:n_agents]
    return [(int(cell), int(rng.integers(CITY.n_destinations))) for cell in cells]


class Agent:
    def __init__(self, agent_id, destination, cell, spawn_tick):
        self.id, self.destination, self.cell, self.spawn_tick = agent_id, destination, cell, spawn_tick
        self.previous_cell = cell
        self.wait_ticks = 0
        self.stuck_ticks = 0
        self.last_decision = None


class Simulation:
    def __init__(self, arrivals, decision_period=DEFAULT_DECISION_PERIOD, max_agents_on_map=None):
        self.city = CITY
        self.max_agents_on_map = max_agents_on_map or CITY.scenario.max_agents_on_map
        self.arrivals, self.next_arrival = arrivals, 0
        self.decision_period = decision_period
        self.time = 0
        self.agents = {}
        self.agents_per_cell = Counter()
        self.signal = [CLOSED] * CITY.n_intersections
        self.all_red_ticks = [0] * CITY.n_intersections
        self.waiting_since_decision = [0.0] * CITY.n_intersections
        self.next_agent_id = 0
        self.trips = []
        self.hole_hits = 0
        self.light_return = 0.0
        self.router_return = 0.0

    def set_signal(self, intersection, state):
        switching_axis = state != self.signal[intersection] and CLOSED not in (self.signal[intersection], state)
        if switching_axis:
            self.all_red_ticks[intersection] = 1
        self.signal[intersection] = state

    def waiting_by_axis(self, intersection):
        count = [sum(self.agents_per_cell[cell] for cell in arm) for arm in self.city.arms[intersection]]
        return count[NORTH] + count[SOUTH], count[EAST] + count[WEST]

    def longest_waits(self):
        longest = [0] * self.city.n_intersections
        for agent in self.agents.values():
            intersection = self.city.arm_head.get(agent.cell)
            if intersection is not None:
                longest[intersection] = max(longest[intersection], agent.stuck_ticks)
        return longest

    def decision_cost(self, intersection, longest_wait, fairness_weight):
        cost = self.waiting_since_decision[intersection] / self.decision_period + fairness_weight * longest_wait
        self.light_return -= cost
        return cost

    def is_done(self):
        everything_arrived = self.next_arrival >= len(self.arrivals) and not self.agents
        return everything_arrived or self.time >= self.city.scenario.max_ticks

    def tick(self, router):
        self.spawn_due_agents()
        directions = self._choose_moves(router)
        moves, hit_hole = self._check_obstacles(directions)
        arrived = self._move(moves)
        self._reward(router, directions, moves, hit_hole, arrived)
        self.time += 1
        for agent in arrived:
            self._remove(agent)
            self.trips.append((self.time - agent.spawn_tick, agent.wait_ticks))
        self.all_red_ticks = [max(0, ticks - 1) for ticks in self.all_red_ticks]

    def _choose_moves(self, router):
        agents = list(self.agents.values())
        for agent in agents:
            agent.previous_cell = agent.cell
        return dict(zip(agents, router.choose_moves(agents)))

    def spawn_due_agents(self):
        while self.next_arrival < len(self.arrivals) and self.arrivals[self.next_arrival].tick <= self.time:
            if len(self.agents) >= self.max_agents_on_map:
                break
            arrival = self.arrivals[self.next_arrival]
            agent = Agent(self.next_agent_id, arrival.destination, arrival.cell, self.time)
            self.next_agent_id += 1
            self.agents[agent.id] = agent
            self.agents_per_cell[agent.cell] += 1
            self.next_arrival += 1

    def _check_obstacles(self, directions):
        moves, hit_hole = {}, []
        for agent, direction in directions.items():
            target = self.city.neighbor[agent.cell][direction]
            intersection = self.city.intersection_index.get(target)
            if target in self.city.hole_cells:
                hit_hole.append(agent)
            elif intersection is not None and self.is_red(intersection, direction):
                self.waiting_since_decision[intersection] += 1
            else:
                moves[agent] = target
        return moves, hit_hole

    def is_red(self, intersection, direction):
        axis_with_green = EAST_WEST if direction % 2 == 1 else NORTH_SOUTH
        return self.all_red_ticks[intersection] > 0 or self.signal[intersection] != axis_with_green

    def _move(self, moves):
        arrived = []
        for agent, target in moves.items():
            self.agents_per_cell[agent.cell] -= 1
            agent.cell = target
            self.agents_per_cell[target] += 1
            if target == self.city.door_cell[agent.destination]:
                arrived.append(agent)
        return arrived

    def _reward(self, router, directions, moves, hit_hole, arrived):
        hit_hole_agents, arrived_agents = set(hit_hole), set(arrived)
        self.hole_hits += len(hit_hole_agents)
        for agent in directions:
            agent.wait_ticks += agent not in moves
            agent.stuck_ticks = 0 if agent in moves else agent.stuck_ticks + 1
            closer_by = (self.city.distance_to_door(agent.previous_cell, agent.destination)
                         - self.city.distance_to_door(agent.cell, agent.destination))
            reward = STEP_REWARD + DISTANCE_REWARD * closer_by - (HOLE_PENALTY if agent in hit_hole_agents else 0)
            self.router_return += reward
            router.learn(agent, reward, agent in arrived_agents)

    def _remove(self, agent):
        self.agents_per_cell[agent.cell] -= 1
        del self.agents[agent.id]

    def mean_trip_ticks(self):
        return float(np.mean([trip for trip, _ in self.trips])) if self.trips else float("nan")

    def mean_wait_ticks(self):
        return float(np.mean([wait for _, wait in self.trips])) if self.trips else 0.0
