from collections import defaultdict

import numpy as np


class QLight:

    def __init__(self, k=3, bins=4, alpha=0.1, gamma=0.9, seed=0):
        self.k, self.bins, self.alpha, self.gamma = k, bins, alpha, gamma
        self.rng = np.random.default_rng(seed)
        self.Q = defaultdict(lambda: np.zeros(2))
        self.eps, self.samples, self.pca, self.prev = 0.0, [], None, {}

    @property
    def ready(self):
        return self.pca is not None

    def fit_pca(self):
        X = np.array(self.samples, float)
        mu, sd = X.mean(0), X.std(0)
        sd = np.where(sd < 1e-9, 1.0, sd)
        Z = (X - mu) / sd
        W = np.linalg.svd(Z, full_matrices=False)[2][: self.k].T
        edges = [np.unique(np.quantile(Z @ W[:, i], np.linspace(0, 1, self.bins + 1)[1:-1])) for i in range(self.k)]
        self.pca = (mu, sd, W, edges)
        self.samples = []

    def _state(self, env, v):
        mu, sd, W, edges = self.pca
        z = ((np.array(env.features(v), float) - mu) / sd) @ W
        return tuple(int(np.searchsorted(edges[i], z[i])) for i in range(self.k))

    def start_episode(self):
        self.prev = {}

    def decide(self, env, train):
        for v in range(env.city.n_int):
            ns, ew = env.demand(v)
            if ns + ew == 0:            # sem atividade: o semáforo fica fechado (todos vermelhos)
                env.set_phase(v, -1)
                env.queue_acc[v] = 0.0
                self.prev.pop(v, None)
                continue
            if not self.ready:
                self.samples.append(env.features(v))
                env.set_phase(v, int(self.rng.integers(2)))
                continue
            s = self._state(env, v)
            if train and v in self.prev:
                ps, pa = self.prev[v]
                r = -env.queue_acc[v] / env.min_pt
                self.Q[ps][pa] += self.alpha * (r + self.gamma * self.Q[s].max() - self.Q[ps][pa])
            env.queue_acc[v] = 0.0
            q = self.Q[s]
            if train and self.rng.random() < self.eps:
                a = int(self.rng.integers(2))
            elif q[0] == q[1]:          # empate: mantém o eixo aberto ou abre o de maior demanda
                a = env.phase[v] if env.phase[v] >= 0 else int(ns > ew)
            else:
                a = int(np.argmax(q))
            self.prev[v] = (s, a)
            env.set_phase(v, a)
