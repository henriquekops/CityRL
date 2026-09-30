import random

N, E, S, W = range(4)
DRC = [(-1, 0), (0, 1), (1, 0), (0, -1)]
OPP = [2, 3, 0, 1]

M = 4             # ruas por direção = cruzamentos por lado
L = 4             # células de cada trecho entre dois cruzamentos
STEP = L + 1
SIZE = (M - 1) * STEP + 1
# casas de destino A, B, C, D (linha, coluna); cada uma tem uma porta numa rua vizinha
DEST_HOUSES = [(1, 2), (4, 8), (11, 12), (14, 7)]
HOLE_RATE = 0.04    # fração das células de rua com buraco (taxa baixa)
HOLE_SEED = 7       # os buracos são sorteados uma vez com esta semente: o mapa é sempre o mesmo


def is_road(r, c):
    return r % STEP == 0 or c % STEP == 0


class City:
    def __init__(self):
        self.n_int = M * M
        self.xy = [(i * STEP, j * STEP) for i in range(M) for j in range(M)]
        self.nbr = [[-1] * 4 for _ in range(self.n_int)]
        for i in range(M):
            for j in range(M):
                for d, (dr, dc) in enumerate(DRC):
                    if 0 <= i + dr < M and 0 <= j + dc < M:
                        self.nbr[i * M + j][d] = (i + dr) * M + j + dc

        self.lane_id, self.lane_from, self.lane_to, self.lane_dir = {}, [], [], []
        self.out_lane = [[-1] * 4 for _ in range(self.n_int)]
        self.in_lane = [[-1] * 4 for _ in range(self.n_int)]
        for v in range(self.n_int):
            for d in range(4):
                w = self.nbr[v][d]
                if w >= 0:
                    self.out_lane[v][d] = self.lane_id[(v, w)] = len(self.lane_from)
                    self.lane_from.append(v)
                    self.lane_to.append(w)
                    self.lane_dir.append(d)
        for v in range(self.n_int):
            for d in range(4):
                if self.nbr[v][d] >= 0:
                    self.in_lane[v][d] = self.lane_id[(self.nbr[v][d], v)]
        self.n_lanes = len(self.lane_from)

        self.segs = sorted(k for k in self.lane_id if k[0] < k[1])   # trechos (u < v)
        self.seg_of_lane = [0] * self.n_lanes
        for s, (u, v) in enumerate(self.segs):
            self.seg_of_lane[self.lane_id[(u, v)]] = self.seg_of_lane[self.lane_id[(v, u)]] = s

        self.houses = self._houses()
        self.dest_cell = [self._door_cells(self.houses[rc]) for rc in DEST_HOUSES]
        self.border_lanes = [l for l in range(self.n_lanes) if self._on_border(l)]
        self.holes = self._make_holes()          # [(trecho, posição)]
        self.hole_cells = [self.cell_xy(self.lane_id[self.segs[s]], p) for s, p in self.holes]
        self.blocked = {self.lane_id[self.segs[s]] * L + p for s, p in self.holes} | \
                       {self.lane_id[self.segs[s][::-1]] * L + (L - 1 - p) for s, p in self.holes}

    def _make_holes(self):
        door_segs = {self.houses[rc][0] for rc in DEST_HOUSES}
        eligible = [s for s, (u, v) in enumerate(self.segs)
                    if self.lane_id[(u, v)] not in self.border_lanes and s not in door_segs]
        n = round(HOLE_RATE * len(self.segs) * L)
        rng = random.Random(HOLE_SEED)
        while True:
            segs = rng.sample(eligible, n)
            if self._connected(set(range(len(self.segs))) - set(segs)):
                return [(s, rng.randrange(L)) for s in sorted(segs)]

    def _connected(self, open_segs):
        seen, todo = {0}, [0]
        while todo:
            v = todo.pop()
            for s in open_segs:
                u, w = self.segs[s]
                if v in (u, w) and (u if v == w else w) not in seen:
                    seen.add(u if v == w else w)
                    todo.append(u if v == w else w)
        return len(seen) == self.n_int

    def _on_border(self, lane):
        (r1, c1), (r2, c2) = self.xy[self.lane_from[lane]], self.xy[self.lane_to[lane]]
        return r1 == r2 in (0, SIZE - 1) or c1 == c2 in (0, SIZE - 1)

    def cell_xy(self, lane, pos):
        r, c = self.xy[self.lane_from[lane]]
        dr, dc = DRC[self.lane_dir[lane]]
        return r + dr * (pos + 1), c + dc * (pos + 1)

    def road_cell(self, r, c):
        i, j = r // STEP, c // STEP
        if r % STEP == 0 and c % STEP != 0:
            return self.segs.index((i * M + j, i * M + j + 1)), c - j * STEP - 1
        if c % STEP == 0 and r % STEP != 0:
            return self.segs.index((i * M + j, (i + 1) * M + j)), r - i * STEP - 1
        return None

    def _houses(self):
        houses = {}
        for r in range(SIZE):
            for c in range(SIZE):
                if not is_road(r, c):
                    for dr, dc in DRC:
                        door = self.road_cell(r + dr, c + dc)
                        if door:
                            houses[(r, c)] = door
                            break
        return houses

    def _door_cells(self, door):
        seg, p = door
        u = self.segs[seg][0]
        return [-1 if self.seg_of_lane[l] != seg else (p if self.lane_from[l] == u else L - 1 - p)
                for l in range(self.n_lanes)]

    def street_spawn(self, r, c, fx, fy):
        door = self.road_cell(r, c)
        if door is None:
            return None
        seg, p = door
        u, v = self.segs[seg]
        forward = fy > 0.5 if r % STEP == 0 else fx < 0.5      # leste ou sul = sentido u -> v
        lane = self.lane_id[(u, v) if forward else (v, u)]
        pos = self._door_cells((seg, p))[lane]
        return None if lane * L + pos in self.blocked else (lane, pos)


CITY = City()
assert all(rc in CITY.houses for rc in DEST_HOUSES), "destino deve ser uma casa com porta"
