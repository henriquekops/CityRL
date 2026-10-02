"""Fixed city map: a gridworld with 3 horizontal and 3 vertical streets (4 blocks), a traffic light at each
crossing, houses (the destinations) and one hole drawn at random each time the program starts.

The map is a grid of cells and every cell is identified by one number: cell = row * GRID_SIZE + col.
A street cell is where an agent can stand; a cell holds at most one agent."""
import random

NORTH, EAST, SOUTH, WEST = range(4)
DIRECTION_DELTAS = [(-1, 0), (0, 1), (1, 0), (0, -1)]       # (row, col) step of each direction

STREETS_PER_SIDE = 3
BLOCK_SIZE = 4                                               # house cells between two parallel streets
ROAD_SPACING = BLOCK_SIZE + 1                                # cells between two parallel streets
GRID_SIZE = (STREETS_PER_SIDE - 1) * ROAD_SPACING + 1

DESTINATION_NAMES = "ABCD"
DESTINATION_HOUSES = [(1, 2), (2, 9), (9, 7), (7, 1)]        # (row, col) of A, B, C, D: one house per block
HOLE_SEED = None                                             # None: a new random hole at every start; an int fixes it


def cell_of(row, col):
    return row * GRID_SIZE + col


def position_of(cell):
    return divmod(cell, GRID_SIZE)


def is_street(row, col):
    return row % ROAD_SPACING == 0 or col % ROAD_SPACING == 0


def is_intersection(row, col):
    return row % ROAD_SPACING == 0 and col % ROAD_SPACING == 0


def street_neighbor(row, col, direction):
    """Cell next to (row, col) in a direction if it is a street cell, else -1."""
    d_row, d_col = DIRECTION_DELTAS[direction]
    row, col = row + d_row, col + d_col
    inside = 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE
    return cell_of(row, col) if inside and is_street(row, col) else -1


class City:
    def __init__(self):
        cells = range(GRID_SIZE * GRID_SIZE)
        self.street_cells = [cell for cell in cells if is_street(*position_of(cell))]
        self.intersections = [cell for cell in self.street_cells if is_intersection(*position_of(cell))]
        self.intersection_index = {cell: index for index, cell in enumerate(self.intersections)}
        self.n_intersections = len(self.intersections)
        self.neighbor = {cell: [street_neighbor(*position_of(cell), d) for d in range(4)] for cell in self.street_cells}
        self.valid_moves = {cell: [other >= 0 for other in others] for cell, others in self.neighbor.items()}
        self.arms = [[self._arm(cell, direction) for direction in range(4)] for cell in self.intersections]
        self.door_cell = [self._door_of(house) for house in DESTINATION_HOUSES]
        self.hole_cell = self._random_hole_cell()
        self.spawn_cells = [cell for cell in self.street_cells if self._is_on_border(cell)
                            and cell not in self.intersection_index and cell != self.hole_cell]
        self.hole = position_of(self.hole_cell)

    def _random_hole_cell(self):
        """Any street cell except intersections (traffic lights) and destination doors. A single hole never
        disconnects the map: every street cell that remains still reaches an intersection."""
        candidates = [cell for cell in self.street_cells
                      if cell not in self.intersection_index and cell not in self.door_cell]
        return random.Random(HOLE_SEED).choice(candidates)

    def _arm(self, intersection_cell, direction):
        """The street cells leaving an intersection in one direction, up to the next intersection."""
        arm, cell = [], intersection_cell
        while len(arm) < BLOCK_SIZE and self.neighbor[cell][direction] >= 0:
            cell = self.neighbor[cell][direction]
            arm.append(cell)
        return arm

    @staticmethod
    def _door_of(house):
        """A house's door is the street cell next to it; reaching it means arriving."""
        for direction in range(4):
            door = street_neighbor(*house, direction)
            if door >= 0:
                return door
        raise ValueError(f"house {house} is not next to a street")

    @staticmethod
    def _is_on_border(cell):
        row, col = position_of(cell)
        return row in (0, GRID_SIZE - 1) or col in (0, GRID_SIZE - 1)

    def distance_to_door(self, cell, destination):
        """Straight-line distance over the grid (|rows| + |cols|) from a cell to a destination's door. It ignores
        blocks and the hole, so it points the way without giving the route away."""
        (row, col), (door_row, door_col) = position_of(cell), position_of(self.door_cell[destination])
        return abs(row - door_row) + abs(col - door_col)

    def can_place_agent(self, cell):
        """Agents may start on any street cell except intersections (traffic lights) and the hole."""
        return cell in self.neighbor and cell not in self.intersection_index and cell != self.hole_cell


CITY = City()
