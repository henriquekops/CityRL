import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from .city import DEST_HOUSES
from .city import CITY, HOLE_RATE
from .env import GRIDLOCK_PENALTY, Env
from .lights import QLight
from .routing import QRouter

COLLECT_EPOCHS = 10        # épocas de coleta de estados para ajustar o PCA (não contam como treino)
RESULTS = Path("results/training.json")


@dataclass
class Hyper:
    light_alpha: float = 0.1      # semáforo: taxa de aprendizado
    light_gamma: float = 0.9      # semáforo: desconto
    light_eps: float = 0.3        # semáforo: ε inicial (decai até 0,02)
    pca_k: int = 4                # semáforo: componentes principais
    pca_bins: int = 6             # semáforo: faixas por componente
    min_pt: int = 3               # semáforo: ticks entre decisões
    router_alpha: float = 0.1     # veículo: taxa de aprendizado
    router_gamma: float = 0.99    # veículo: desconto
    router_eps: float = 0.05      # veículo: ε inicial (decai até 0,02)
    propagation: bool = True      # veículo: propagação retroativa
    omega: float = 0.05           # veículo: limite para parar de propagar


class Trainer:
    def __init__(self, hp=None):
        self.hp = hp or Hyper()
        self.light = QLight(self.hp.pca_k, self.hp.pca_bins, self.hp.light_alpha, self.hp.light_gamma)
        self.router = QRouter(self.hp.router_alpha, self.hp.router_gamma,
                              self.hp.omega if self.hp.propagation else None)
        self.history = []
        self.run_id = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @property
    def epoch(self):
        return len(self.history)

    def run_epoch(self, env, train, watch=None):
        self.router.train = train
        self.light.start_episode()
        while not env.done():
            self.step(env, train)
            if watch:
                watch(env, self.epoch + 1)

    def step(self, env, train=False):
        if env.t % env.min_pt == 0:
            self.light.decide(env, train)
        env.tick(self.router)

    def train(self, epochs, n_agents, on_epoch=None, should_stop=lambda: False, watch=None):
        hp = self.hp
        if not self.light.ready:
            for i in range(COLLECT_EPOCHS):
                self.run_epoch(Env(n_agents, seed=90_000 + i, min_pt=hp.min_pt), train=False)
            self.light.fit_pca()
        for i in range(epochs):
            if should_stop():
                break
            decay = 1 - i / (0.7 * epochs)
            self.light.eps = max(0.02, hp.light_eps * decay)
            self.router.eps = max(0.02, hp.router_eps * decay)
            env = Env(n_agents, seed=self.epoch, min_pt=hp.min_pt)   # mesma semente por época em toda execução
            self.run_epoch(env, train=True, watch=watch)
            self.history.append(dict(
                epoch=self.epoch + 1, n_agents=n_agents, arrived=len(env.finished), gridlocked=env.teleported,
                mean_trip_ticks=round(env.mean_trip(), 2),
                mean_wait_ticks=round(sum(f[1] for f in env.finished) / max(1, len(env.finished)), 2),
                makespan_ticks=env.t, cumulative_ticks=sum(h["makespan_ticks"] for h in self.history) + env.t,
                light_eps=round(self.light.eps, 3), router_eps=round(self.router.eps, 3)))
            if on_epoch:
                on_epoch(self.history[-1], epochs)
        self.save()

    def agents_env(self, agents):
        self.router.train = False
        self.light.start_episode()
        return Env(agents=agents, min_pt=self.hp.min_pt)

    def save(self, path=RESULTS):
        path = Path(path)
        path.parent.mkdir(exist_ok=True)
        try:
            runs = [r for r in json.loads(path.read_text())["runs"] if r["id"] != self.run_id]
        except (OSError, ValueError, KeyError):
            runs = []
        last = [e["mean_trip_ticks"] for e in self.history[-10:]]
        runs.append(dict(id=self.run_id, hyperparameters=asdict(self.hp),
                         environment=dict(hole_rate=HOLE_RATE, holes_cells=[list(h) for h in CITY.hole_cells],
                                          gridlock_penalty=GRIDLOCK_PENALTY),
                         epochs_trained=self.epoch,
                         total_ticks_trained=self.history[-1]["cumulative_ticks"] if self.history else 0,
                         final_mean_trip_ticks=round(sum(last) / len(last), 2) if last else None,
                         epochs=self.history))
        path.write_text(json.dumps(dict(
            destinations=[dict(name="ABCD"[k], house=list(h)) for k, h in enumerate(DEST_HOUSES)],
            runs=runs), indent=1, ensure_ascii=False))
