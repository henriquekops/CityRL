import numpy as np

from .city import CITY, DEST_HOUSES


class QRouter:

    def __init__(self, alpha=0.3, gamma=0.99, omega=0.05, seed=0):
        self.alpha, self.gamma, self.omega = alpha, gamma, omega
        self.Q = np.zeros((len(DEST_HOUSES), CITY.n_int, 4))
        self.rng = np.random.default_rng(seed)
        self.eps, self.train = 0.0, False

    def _pick(self, q, mask):
        q = np.where(mask, q, -np.inf)
        return int(self.rng.choice(np.flatnonzero(q >= q.max() - 1e-12)))

    def _target(self, k, rec):
        _, _, r, s2, mask2, dt, terminal = rec
        return r if terminal else r + self.gamma ** dt * np.max(np.where(mask2, self.Q[k, s2], -np.inf))

    def _learn(self, veh, s2, mask2, t, terminal, penalty=0.0):
        s, a, t0 = veh.prev
        k, dt = veh.dest, t - t0
        veh.traj.append((s, a, -float(dt) - penalty, s2, mask2, dt, terminal))
        for i, rec in enumerate(reversed(veh.traj)):
            delta = self._target(k, rec) - self.Q[k, rec[0], rec[1]]
            if i > 0 and (self.omega is None or abs(delta) < self.omega):
                break
            self.Q[k, rec[0], rec[1]] += self.alpha * delta

    def decide(self, veh, v, mask, t):
        if self.train and veh.prev is not None:
            self._learn(veh, v, mask, t, terminal=False)
        if self.train and self.rng.random() < self.eps:
            a = int(self.rng.choice(np.flatnonzero(mask)))
        else:
            a = self._pick(self.Q[veh.dest, v], mask)
        veh.prev = (v, a, t)
        return a

    def arrive(self, veh, t, penalty=0.0):
        """Fim da viagem: chegada ao destino ou remoção por ficar preso (`penalty` > 0, recompensa negativa)."""
        if self.train and veh.prev is not None:
            self._learn(veh, -1, None, t, terminal=True, penalty=penalty)
        veh.prev, veh.traj = None, []
