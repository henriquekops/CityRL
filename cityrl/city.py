import random
from dataclasses import dataclass

NORTH, EAST, SOUTH, WEST = range(4)
DIRECTION_DELTAS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

BLOCK_SIZE = 4
ROAD_SPACING = BLOCK_SIZE + 1


@dataclass(frozen=True)
class Scenario:
    name: str
    streets_per_side: int
    n_destinations: int
    n_holes: int
    spawn_window_ticks: int
    max_agents_on_map: int
    max_ticks: int
    default_agents: int
    houses: tuple = None


SCENARIOS = {scenario.name: scenario for scenario in (
    Scenario("9 crossings (11x11)", streets_per_side=3, n_destinations=4, n_holes=1, spawn_window_ticks=100,
             max_agents_on_map=30, max_ticks=1500, default_agents=300, houses=((1, 2), (2, 9), (9, 7), (7, 1))),
    Scenario("16 crossings (16x16)", streets_per_side=4, n_destinations=6, n_holes=2, spawn_window_ticks=150,
             max_agents_on_map=90, max_ticks=2000, default_agents=900),
    Scenario("36 crossings (26x26)", streets_per_side=6, n_destinations=8, n_holes=4, spawn_window_ticks=200,
             max_agents_on_map=220, max_ticks=3000, default_agents=2200),
    Scenario("64 crossings (36x36)", streets_per_side=8, n_destinations=10, n_holes=6, spawn_window_ticks=250,
             max_agents_on_map=400, max_ticks=4000, default_agents=4000),
)}
DEFAULT_SCENARIO = "9 crossings (11x11)"
HOUSE_SEED = 0
HOLE_SEED = None


def cell_of(row, col):
    return row * CITY.grid_size + col


def position_of(cell):
    return divmod(cell, CITY.grid_size)


def is_street(row, col):
    return row % ROAD_SPACING == 0 or col % ROAD_SPACING == 0


def is_intersection(row, col):
    return row % ROAD_SPACING == 0 and col % ROAD_SPACING == 0


class City:

    def configure(self, scenario_name, hole_seed=HOLE_SEED):
        self.scenario = scenario = SCENARIOS[scenario_name]
        self.grid_size = (scenario.streets_per_side - 1) * ROAD_SPACING + 1
        cells = range(self.grid_size * self.grid_size)
        self.street_cells = [cell for cell in cells if is_street(*position_of(cell))]
        self.intersections = [cell for cell in self.street_cells if is_intersection(*position_of(cell))]
        self.intersection_index = {cell: index for index, cell in enumerate(self.intersections)}
        self.n_intersections = len(self.intersections)
        self.neighbor = {cell: [self._street_neighbor(cell, d) for d in range(4)] for cell in self.street_cells}
        self.valid_moves = {cell: [other >= 0 for other in others] for cell, others in self.neighbor.items()}
        self.arms = [[self._arm(cell, direction) for direction in range(4)] for cell in self.intersections]
        self.arm_head = {arm[0]: intersection for intersection, arms in enumerate(self.arms) for arm in arms if arm}
        self.neighbor_intersection = [[self._next_intersection(intersection, direction) for direction in range(4)]
                                      for intersection in range(self.n_intersections)]
        self.destination_houses = list(scenario.houses or self._random_houses(scenario.n_destinations))
        self.n_destinations = len(self.destination_houses)
        self.destination_names = [chr(ord("A") + index) for index in range(self.n_destinations)]
        self.door_cell = [self._door_of(house) for house in self.destination_houses]
        self.place_holes(hole_seed)

    def place_holes(self, seed):
        rng = random.Random(seed)
        candidates = [cell for cell in self.street_cells if cell not in self.intersection_index
                      and cell not in self.door_cell and not set(self.neighbor[cell]) & set(self.door_cell)]
        rng.shuffle(candidates)
        holes, used_crossings = [], set()
        for cell in candidates:
            ends = self._segment_ends(cell)
            if len(holes) < self.scenario.n_holes and not ends & used_crossings:
                holes.append(cell)
                used_crossings |= ends
        self.set_holes([position_of(cell) for cell in holes])

    def set_holes(self, positions):
        self.holes = [tuple(position) for position in positions]
        self.hole_cells = {cell_of(*position) for position in self.holes}
        self.spawn_cells = self.agent_start_cells()

    @property
    def max_spawned_agents(self):
        return len(self.agent_start_cells())

    def _segment_ends(self, cell):
        row, col = position_of(cell)
        if row % ROAD_SPACING == 0:
            left = col // ROAD_SPACING * ROAD_SPACING
            return {cell_of(row, left), cell_of(row, left + ROAD_SPACING)}
        top = row // ROAD_SPACING * ROAD_SPACING
        return {cell_of(top, col), cell_of(top + ROAD_SPACING, col)}

    def _street_neighbor(self, cell, direction):
        row, col = position_of(cell)
        d_row, d_col = DIRECTION_DELTAS[direction]
        row, col = row + d_row, col + d_col
        inside = 0 <= row < self.grid_size and 0 <= col < self.grid_size
        return cell_of(row, col) if inside and is_street(row, col) else -1

    def _arm(self, intersection_cell, direction):
        arm, cell = [], intersection_cell
        while len(arm) < BLOCK_SIZE and self.neighbor[cell][direction] >= 0:
            cell = self.neighbor[cell][direction]
            arm.append(cell)
        return arm

    def _next_intersection(self, intersection, direction):
        arm = self.arms[intersection][direction]
        following = self.neighbor[arm[-1]][direction] if arm else -1
        return self.intersection_index.get(following, -1)

    def _random_houses(self, count):
        by_block = {}
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if not is_street(row, col) and self._door_of((row, col), required=False) >= 0:
                    by_block.setdefault((row // ROAD_SPACING, col // ROAD_SPACING), []).append((row, col))
        rng = random.Random(HOUSE_SEED)
        blocks = sorted(by_block)
        rng.shuffle(blocks)
        return [rng.choice(by_block[block]) for block in blocks[:count]]

    def _door_of(self, house, required=True):
        row, col = house
        for d_row, d_col in DIRECTION_DELTAS:
            door_row, door_col = row + d_row, col + d_col
            inside = 0 <= door_row < self.grid_size and 0 <= door_col < self.grid_size
            if inside and is_street(door_row, door_col) and not is_intersection(door_row, door_col):
                return cell_of(door_row, door_col)
        if required:
            raise ValueError(f"house {house} is not next to a street")
        return -1

    def distance_to_door(self, cell, destination):
        (row, col), (door_row, door_col) = position_of(cell), position_of(self.door_cell[destination])
        return abs(row - door_row) + abs(col - door_col)

    def can_place_agent(self, cell):
        return cell in self.neighbor and cell not in self.intersection_index and cell not in self.hole_cells

    def agent_start_cells(self):
        doors = set(self.door_cell)
        return [cell for cell in self.street_cells if self.can_place_agent(cell) and cell not in doors]


CITY = City()
CITY.configure(DEFAULT_SCENARIO)
