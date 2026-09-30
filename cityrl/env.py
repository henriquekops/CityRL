import numpy as np

from .city import CITY, DEST_HOUSES, OPP, L

SPAWN_WINDOW = 300   # os veículos de uma época entram ao longo dos primeiros ticks
MIN_PT = 5           # padrão: o semáforo decide a cada MIN_PT ticks
GRIDLOCK_PENALTY = 100  # custo extra (recompensa negativa) para quem fica preso até ser removido
TELEPORT = 60        # veículo preso por tanto tempo é removido (gridlock), como no SUMO


class Vehicle:
    def __init__(self, vid, dest, lane, pos, t0):
        self.id, self.dest, self.lane, self.pos, self.t0 = vid, dest, lane, pos, t0
        self.next_lane, self.wait, self.stuck = -1, 0, 0
        self.prev, self.traj = None, []     # estado do roteador (última decisão e trajetória)


class Env:

    def __init__(self, n_agents=300, seed=0, agents=None, max_ticks=1500, max_active=100, min_pt=MIN_PT):
        self.city = CITY
        self.max_ticks, self.max_active, self.min_pt = max_ticks, max_active, min_pt
        c = self.city
        self.t = 0
        self.occ = [None] * (c.n_lanes * L)
        self.active = {}
        self.phase = [-1] * c.n_int       # -1: fechado (sem demanda), 0: leste-oeste verde, 1: norte-sul verde
        self.red = [0] * c.n_int          # 1 tick de "todos vermelhos" após trocar de fase
        self.streak = [0] * c.n_int
        self.queue_acc = [0.0] * c.n_int
        self.rng = np.random.default_rng(seed + 10_000)
        self.blocked = c.blocked                                    # células com buraco: ninguém entra
        self.finished, self.teleported, self.next_id, self.sched_i = [], 0, 0, 0
        if agents:
            self.schedule = [(0, dest, lane, pos) for lane, pos, dest in agents]
        else:
            rng = np.random.default_rng(seed)
            self.schedule = []
            for t in sorted(rng.integers(0, SPAWN_WINDOW, n_agents)):
                dest = int(rng.integers(len(DEST_HOUSES)))
                lane = int(rng.choice(c.border_lanes))
                if c.dest_cell[dest][lane] != 0:
                    self.schedule.append((int(t), dest, lane, 0))

    def set_phase(self, v, a):
        if a != self.phase[v]:
            if self.phase[v] >= 0 and a >= 0:   # trocar de eixo custa 1 tick de "todos vermelhos"
                self.red[v] = 1
            self.phase[v], self.streak[v] = a, 0
        else:
            self.streak[v] += 1

    def features(self, v):
        c, f = self.city, []
        for d in range(4):
            l = c.in_lane[v][d]
            f += [0] * L if l < 0 else [int(self.occ[l * L + p] is not None) for p in range(L - 1, -1, -1)]
        for d in range(4):
            l = c.out_lane[v][d]
            f.append(0 if l < 0 else sum(self.occ[l * L + p] is not None for p in range(L)))
        return f + [self.phase[v], min(self.streak[v], 3)]

    def demand(self, v):
        f = self.features(v)
        return sum(f[0:4]) + sum(f[8:12]), sum(f[4:8]) + sum(f[12:16])

    def done(self):
        return (self.sched_i >= len(self.schedule) and not self.active) or self.t >= self.max_ticks

    def _spawn(self):
        while self.sched_i < len(self.schedule) and self.schedule[self.sched_i][0] <= self.t:
            _, dest, lane, pos = self.schedule[self.sched_i]
            if len(self.active) >= self.max_active or self.occ[lane * L + pos] is not None:
                break
            veh = Vehicle(self.next_id, dest, lane, pos, self.t)
            self.next_id += 1
            self.occ[lane * L + pos] = veh
            self.active[veh.id] = veh
            self.sched_i += 1

    def _choose_exit(self, veh, v, router):
        c = self.city
        mask = [c.nbr[v][d] >= 0 and d != OPP[c.lane_dir[veh.lane]] for d in range(4)]
        veh.next_lane = c.out_lane[v][router.decide(veh, v, mask, self.t)]

    def tick(self, router):
        c = self.city
        self._spawn()
        cand, stopped = {}, []
        for veh in self.active.values():
            if veh.pos < L - 1:
                tgt = veh.lane * L + veh.pos + 1
                if tgt in self.blocked:      # buraco à frente: fica parado
                    stopped.append(veh)
                else:
                    cand[veh] = tgt
                continue
            v = c.lane_to[veh.lane]
            if veh.next_lane < 0:
                self._choose_exit(veh, v, router)
            east_west = c.lane_dir[veh.lane] % 2 == 1
            if self.red[v] == 0 and self.phase[v] == (0 if east_west else 1) and veh.next_lane * L not in self.blocked:
                cand[veh] = veh.next_lane * L
            else:
                stopped.append(veh)

        prio = {veh: self.rng.random() for veh in cand}
        movers, changed = dict(cand), True
        while changed:
            changed, best = False, {}
            for veh, tgt in movers.items():
                if tgt not in best or prio[veh] > prio[best[tgt]]:
                    best[tgt] = veh
            for veh, tgt in list(movers.items()):
                occupant = self.occ[tgt]
                if best[tgt] is not veh or (occupant is not None and occupant not in movers):
                    del movers[veh]
                    changed = True
        stopped += [v for v in cand if v not in movers]

        for veh in movers:
            self.occ[veh.lane * L + veh.pos] = None
        arrived = []
        for veh, tgt in movers.items():
            if veh.pos == L - 1:
                veh.next_lane = -1
            veh.lane, veh.pos = divmod(tgt, L)
            veh.stuck = 0
            self.occ[tgt] = veh
            if self.city.dest_cell[veh.dest][veh.lane] == veh.pos:
                arrived.append(veh)

        gridlocked = []
        for veh in stopped:
            veh.wait += 1
            veh.stuck += 1
            self.queue_acc[c.lane_to[veh.lane]] += 1
            if veh.stuck >= TELEPORT:
                gridlocked.append(veh)
        self.t += 1
        for veh in arrived:
            router.arrive(veh, self.t)
            self.occ[veh.lane * L + veh.pos] = None
            del self.active[veh.id]
            self.finished.append((self.t - veh.t0, veh.wait))
        for veh in gridlocked:
            router.arrive(veh, self.t, GRIDLOCK_PENALTY)    # o roteador aprende que esse caminho é péssimo
            self.occ[veh.lane * L + veh.pos] = None
            del self.active[veh.id]
            self.teleported += 1
        self.red = [max(0, r - 1) for r in self.red]

    def mean_trip(self):
        return float(np.mean([f[0] for f in self.finished])) if self.finished else float("nan")
