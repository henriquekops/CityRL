"""Discrete-time traffic gridworld. One *tick* is one time step in which every agent tries to move one cell.

Rules: agents do not interact with each other (they may even share a cell). The hole blocks: an agent that tries to
enter it stays where it is and is penalized. An agent that tries to enter an intersection on red simply waits. The
simulator applies the rules and does not learn: a *router* chooses each agent's move and a traffic-light agent sets the
signals.

Reward of an agent in one tick = STEP_REWARD + DISTANCE_REWARD * (cells closer to the destination) - hole penalty.
The distance term is reward shaping: moving away from the goal costs, getting closer pays back."""
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .city import CITY, DESTINATION_HOUSES, EAST, NORTH, SOUTH, WEST

SPAWN_WINDOW_TICKS = 100        # agents of an epoch enter the map during the first ticks
DEFAULT_DECISION_PERIOD = 3     # ticks between two decisions of a traffic light
MAX_TICKS = 1500                # an epoch is cut after this many ticks
MAX_AGENTS_ON_MAP = 30          # entries wait outside while the map is this full
STEP_REWARD = -1                # every tick costs one, so agents want short trips
HOLE_PENALTY = 10               # extra cost of trying to enter the hole
DISTANCE_REWARD = 0.5           # per cell closer to the destination (negative per cell farther)

CLOSED, EAST_WEST, NORTH_SOUTH = -1, 0, 1       # states of a traffic light (CLOSED = all red, no demand)


@dataclass
class Arrival:
    """An agent scheduled to enter the map at `tick`, at `cell`, bound for `destination`."""
    tick: int
    destination: int
    cell: int


def random_arrivals(n_agents, seed):
    """Entries at random border cells and random times, with random destinations (same seed, same demand)."""
    rng = np.random.default_rng(seed)
    arrivals = []
    for tick in sorted(rng.integers(0, SPAWN_WINDOW_TICKS, n_agents)):
        destination = int(rng.integers(len(DESTINATION_HOUSES)))
        cell = int(rng.choice(CITY.spawn_cells))
        if cell != CITY.door_cell[destination]:                     # skip entries that start on the door
            arrivals.append(Arrival(int(tick), destination, cell))
    return arrivals


def chosen_arrivals(agents):
    """All entries at tick 0, for visualization. `agents` is a list of (cell, destination)."""
    return [Arrival(0, destination, cell) for cell, destination in agents]


class Agent:
    def __init__(self, agent_id, destination, cell, spawn_tick):
        self.id, self.destination, self.cell, self.spawn_tick = agent_id, destination, cell, spawn_tick
        self.previous_cell = cell       # where the agent was at the start of the tick
        self.wait_ticks = 0             # ticks spent without moving
        self.last_decision = None       # (cell, direction), set and used by the router
        self.transitions = []           # set and used by the router


class Simulation:
    def __init__(self, arrivals, decision_period=DEFAULT_DECISION_PERIOD):
        self.city = CITY
        self.arrivals, self.next_arrival = arrivals, 0
        self.decision_period = decision_period
        self.time = 0
        self.agents = {}                                                    # agent id -> agent on the map
        self.agents_per_cell = Counter()                                    # cell -> number of agents on it
        self.signal = [CLOSED] * CITY.n_intersections
        self.all_red_ticks = [0] * CITY.n_intersections                     # 1 tick of all-red after switching axis
        self.waiting_since_decision = [0.0] * CITY.n_intersections          # agent-ticks held back by each red light
        self.next_agent_id = 0
        self.trips = []                                                     # (trip_ticks, wait_ticks) of arrivals
        self.hole_hits = 0

    # ---------- state seen by the traffic lights ----------
    def set_signal(self, intersection, state):
        switching_axis = state != self.signal[intersection] and CLOSED not in (self.signal[intersection], state)
        if switching_axis:
            self.all_red_ticks[intersection] = 1
        self.signal[intersection] = state

    def waiting_by_axis(self, intersection):
        """(agents on the north/south arms, agents on the east/west arms) of an intersection."""
        count = [sum(self.agents_per_cell[cell] for cell in arm) for arm in self.city.arms[intersection]]
        return count[NORTH] + count[SOUTH], count[EAST] + count[WEST]

    # ---------- one tick ----------
    def is_done(self):
        everything_arrived = self.next_arrival >= len(self.arrivals) and not self.agents
        return everything_arrived or self.time >= MAX_TICKS

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
        directions = {}
        for agent in self.agents.values():
            agent.previous_cell = agent.cell
            directions[agent] = router.choose_move(agent)
        return directions

    def spawn_due_agents(self):
        while self.next_arrival < len(self.arrivals) and self.arrivals[self.next_arrival].tick <= self.time:
            if len(self.agents) >= MAX_AGENTS_ON_MAP:
                break
            arrival = self.arrivals[self.next_arrival]
            agent = Agent(self.next_agent_id, arrival.destination, arrival.cell, self.time)
            self.next_agent_id += 1
            self.agents[agent.id] = agent
            self.agents_per_cell[agent.cell] += 1
            self.next_arrival += 1

    def _check_obstacles(self, directions):
        """Split the intended moves: ({agent: target cell} that go through, agents that hit the hole).
        An agent facing a red light just waits, and the light is charged for it."""
        moves, hit_hole = {}, []
        for agent, direction in directions.items():
            target = self.city.neighbor[agent.cell][direction]
            intersection = self.city.intersection_index.get(target)
            if target == self.city.hole_cell:
                hit_hole.append(agent)
            elif intersection is not None and self._is_red(intersection, direction):
                self.waiting_since_decision[intersection] += 1
            else:
                moves[agent] = target
        return moves, hit_hole

    def _is_red(self, intersection, direction):
        axis_with_green = EAST_WEST if direction % 2 == 1 else NORTH_SOUTH      # east/west directions are odd
        return self.all_red_ticks[intersection] > 0 or self.signal[intersection] != axis_with_green

    def _move(self, moves):
        """Apply the moves; returns the agents that reached their destination door."""
        arrived = []
        for agent, target in moves.items():
            self.agents_per_cell[agent.cell] -= 1
            agent.cell = target
            self.agents_per_cell[target] += 1
            if target == self.city.door_cell[agent.destination]:
                arrived.append(agent)
        return arrived

    def _reward(self, router, directions, moves, hit_hole, arrived):
        """Give every agent its reward for this tick: step cost, distance shaping, and a penalty for the hole."""
        hit_hole_agents, arrived_agents = set(hit_hole), set(arrived)
        self.hole_hits += len(hit_hole_agents)
        for agent in directions:
            agent.wait_ticks += agent not in moves
            closer_by = (self.city.distance_to_door(agent.previous_cell, agent.destination)
                         - self.city.distance_to_door(agent.cell, agent.destination))
            reward = STEP_REWARD + DISTANCE_REWARD * closer_by - (HOLE_PENALTY if agent in hit_hole_agents else 0)
            router.learn(agent, reward, agent in arrived_agents)

    def _remove(self, agent):
        self.agents_per_cell[agent.cell] -= 1
        del self.agents[agent.id]

    # ---------- results ----------
    def mean_trip_ticks(self):
        return float(np.mean([trip for trip, _ in self.trips])) if self.trips else float("nan")

    def mean_wait_ticks(self):
        return float(np.mean([wait for _, wait in self.trips])) if self.trips else 0.0
