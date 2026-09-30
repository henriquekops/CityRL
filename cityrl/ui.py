import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

from .city import CITY, DEST_HOUSES, DRC, SIZE, is_road
from .trainer import Hyper, Trainer

CELL = 44
EPOCHS = 100      # valor inicial do campo "Épocas"
TICK_MS = 250     # visualização lenta dos agentes
MAX_VIEW = 10     # máximo de agentes na visualização
VIEW_EVERY = 10   # ao visualizar o treino, mostra 1 época a cada VIEW_EVERY
TRAIN_TICK_S = 0.015   # pausa por tick nas épocas mostradas
COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231"]
GREEN, RED = "#2ca02c", "#d62728"
# hiperparâmetros na interface: grupo -> (campo de Hyper, rótulo, mínimo, máximo, passo)
HP_FIELDS = {
    "Semáforo": [("light_alpha", "α", 0.01, 1, 0.05), ("light_gamma", "γ", 0.1, 0.999, 0.05),
                 ("light_eps", "ε inicial", 0.02, 1, 0.05), ("pca_k", "k do PCA", 1, 10, 1),
                 ("pca_bins", "faixas/comp.", 2, 8, 1), ("min_pt", "ticks/decisão", 1, 20, 1)],
    "Veículo": [("router_alpha", "α", 0.01, 1, 0.05), ("router_gamma", "γ", 0.1, 0.999, 0.01),
                ("router_eps", "ε inicial", 0.02, 1, 0.05), ("propagation", "propagação retroativa", None, None, None),
                ("omega", "ω (limite)", 0, 5, 0.05)],
}


class App:
    def __init__(self, root):
        self.root, self.trainer = root, None      # o Trainer nasce no 1º treino, com os hiperparâmetros da tela
        self.state = dict(epoch=0, last=None, start=0, frame=None)     # atualizado pela thread de treino
        self.watching = False
        self.training, self.animating, self.env, self.trails = False, False, None, {}
        self.sel = []      # agentes escolhidos: dict(cell=(r, c), spawn=(faixa, pos), dest=k)
        root.title("CityRL")

        n = SIZE * CELL
        self.canvas = tk.Canvas(root, width=n, height=n, bg="white", highlightthickness=0)
        self.canvas.grid(row=0, column=0, padx=8, pady=8)
        self.canvas.bind("<Button-1>", self.pick_spawn)
        self.draw_map()

        side = ttk.Frame(root)
        side.grid(row=0, column=1, sticky="n", padx=(0, 8), pady=8)
        row = ttk.Frame(side)
        row.pack(anchor="w")
        ttk.Label(row, text="Agentes por época").grid(row=0, column=0, sticky="w")
        ttk.Label(row, text="Épocas").grid(row=0, column=1, sticky="w", padx=(12, 0))
        self.agents, self.epochs = tk.IntVar(value=300), tk.IntVar(value=EPOCHS)
        ttk.Spinbox(row, from_=10, to=1000, increment=50, textvariable=self.agents, width=8).grid(row=1, column=0)
        ttk.Spinbox(row, from_=1, to=10000, increment=50, textvariable=self.epochs, width=8).grid(row=1, column=1, padx=(12, 0))
        self.total = EPOCHS      # total da execução atual
        self.build_hyper_panel(side)
        row = ttk.Frame(side)
        row.pack(fill="x", pady=8)
        self.train_btn = ttk.Button(row, text="Treinar", command=self.train)
        self.train_btn.pack(side="left", expand=True, fill="x")
        self.new_btn = ttk.Button(row, text="Novo treino", command=self.new_run)
        self.new_btn.pack(side="left", expand=True, fill="x")
        self.epoch_lbl = ttk.Label(side, text=f"Época 0 / {self.total}", font=("Helvetica", 14, "bold"))
        self.epoch_lbl.pack(anchor="w")
        self.progress = ttk.Progressbar(side, maximum=self.total)
        self.progress.pack(fill="x", pady=4)
        self.watch_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(side, text=f"Visualizar o treino (1 época a cada {VIEW_EVERY})", variable=self.watch_var,
                        command=lambda: setattr(self, "watching", self.watch_var.get())).pack(anchor="w")
        self.info_lbl = ttk.Label(side, text="", justify="left")
        self.info_lbl.pack(anchor="w")

        ttk.Separator(side).pack(fill="x", pady=10)
        ttk.Label(side, text=f"Visualizar agentes (até {MAX_VIEW})").pack(anchor="w")
        row = ttk.Frame(side)
        row.pack(anchor="w")
        ttk.Label(row, text="Destino").pack(side="left")
        self.dest = tk.StringVar(value="A")
        ttk.Combobox(row, textvariable=self.dest, values=list("ABCD"), state="readonly", width=3).pack(side="left", padx=4)
        self.view_btn = ttk.Button(side, text="Visualizar", command=self.visualize, state="disabled")
        self.view_btn.pack(fill="x", pady=(8, 2))
        ttk.Button(side, text="Limpar agentes", command=self.clear_agents).pack(fill="x")
        self.status = ttk.Label(side, text="", foreground="#555", wraplength=200)
        self.status.pack(anchor="w")
        self.status.config(text="Treine antes de visualizar.")
        self.toggle_agent(0, 7, 0.5, 0.75)   # agente inicial: rua do topo, sentido leste, destino A

    def build_hyper_panel(self, parent):
        defaults = Hyper()
        self.hvars, self.hp_widgets = {}, []
        box = ttk.LabelFrame(parent, text="Hiperparâmetros")
        box.pack(fill="x", pady=(8, 0))
        for col, (group, fields) in enumerate(HP_FIELDS.items()):
            frame = ttk.Frame(box)
            frame.grid(row=0, column=col, sticky="n", padx=6, pady=4)
            ttk.Label(frame, text=group, font=("Helvetica", 11, "bold")).grid(row=0, column=0, columnspan=2, sticky="w")
            for r, (name, label, lo, hi, step) in enumerate(fields, start=1):
                default = getattr(defaults, name)
                if isinstance(default, bool):
                    var = tk.BooleanVar(value=default)
                    w = ttk.Checkbutton(frame, text=label, variable=var)
                    w.grid(row=r, column=0, columnspan=2, sticky="w")
                else:
                    var = tk.IntVar(value=default) if isinstance(default, int) else tk.DoubleVar(value=default)
                    ttk.Label(frame, text=label).grid(row=r, column=0, sticky="w")
                    w = ttk.Spinbox(frame, from_=lo, to=hi, increment=step, textvariable=var, width=6)
                    w.grid(row=r, column=1, padx=(6, 0))
                self.hvars[name] = var
                self.hp_widgets.append(w)

    def read_hyper(self):
        try:
            return Hyper(**{name: var.get() for name, var in self.hvars.items()})
        except tk.TclError:
            messagebox.showerror("Hiperparâmetros", "Há um valor inválido nos hiperparâmetros.")
            return None

    def set_hyper_state(self, enabled):
        for w in self.hp_widgets:
            w.config(state="normal" if enabled else "disabled")

    def draw_map(self):
        cv = self.canvas
        for r in range(SIZE):
            for c in range(SIZE):
                cv.create_rectangle(c * CELL, r * CELL, (c + 1) * CELL, (r + 1) * CELL, outline="white",
                                    fill="#d9d9d9" if is_road(r, c) else "#f3e9d2")
        for (r, c) in CITY.hole_cells:      # buracos: quadrado preto que ocupa a célula toda
            cv.create_rectangle(c * CELL, r * CELL, (c + 1) * CELL, (r + 1) * CELL, fill="black", outline="black")
        for (r, c) in CITY.houses:
            if (r, c) in DEST_HOUSES:
                k = DEST_HOUSES.index((r, c))
                cv.create_rectangle(c * CELL + 2, r * CELL + 2, (c + 1) * CELL - 2, (r + 1) * CELL - 2,
                                    fill=COLORS[k], outline="black")
                cv.create_text((c + .5) * CELL, (r + .5) * CELL, text="ABCD"[k], fill="white",
                               font=("Helvetica", 12, "bold"))

    def draw_dynamic(self, frame):
        cv = self.canvas
        cv.delete("dyn")
        for v, (r, c) in enumerate(CITY.xy):
            x, y, go = (c + .5) * CELL, (r + .5) * CELL, frame["red"][v] == 0
            cv.create_rectangle(x - CELL / 2, y - 4, x + CELL / 2, y + 4, tags="dyn", outline="",
                                fill=GREEN if go and frame["phase"][v] == 0 else RED)
            cv.create_rectangle(x - 4, y - CELL / 2, x + 4, y + CELL / 2, tags="dyn", outline="",
                                fill=GREEN if go and frame["phase"][v] == 1 else RED)
            cv.create_rectangle(x - 4, y - 4, x + 4, y + 4, fill="#333", outline="", tags="dyn")
        for color, pts in self.trails.values():
            if len(pts) > 1:
                cv.create_line(*[q for p in pts for q in p], fill=color, width=3, dash=(2, 3), tags="dyn")
        for x, y, dest, vid in frame["cars"]:
            cv.create_oval(x - 9, y - 9, x + 9, y + 9, fill=COLORS[dest], outline="black", width=2, tags="dyn")
            cv.create_text(x, y, text=str(vid + 1), fill="white", font=("Helvetica", 9, "bold"), tags="dyn")

    @classmethod
    def frame_of(cls, env):
        return dict(phase=list(env.phase), red=list(env.red),
                    cars=[(*cls.car_xy(v), v.dest, v.id) for v in list(env.active.values())])

    @staticmethod
    def lane_xy(lane, pos):
        r, c = CITY.cell_xy(lane, pos)
        dr, dc = DRC[CITY.lane_dir[lane]]
        return (c + .5 - 0.22 * dr) * CELL, (r + .5 + 0.22 * dc) * CELL

    @classmethod
    def car_xy(cls, veh):
        return cls.lane_xy(veh.lane, veh.pos)

    def train(self):
        if self.training:
            return
        if self.trainer is None:  # 1º treino de uma execução: fixa os hiperparâmetros
            hp = self.read_hyper()
            if hp is None:
                return
            self.trainer = Trainer(hp)
            self.set_hyper_state(False)
        self.training = True
        self.total = max(1, int(self.epochs.get()))
        self.progress.config(maximum=self.total)
        self.epoch_lbl.config(text=f"Época 0 / {self.total}")
        self.state.update(epoch=0, last=None, start=self.trainer.epoch, frame=None)   # contador da execução atual
        self.canvas.delete("dyn")
        self.env, self.trails = None, {}
        self.train_btn.config(state="disabled")
        self.new_btn.config(state="disabled")
        self.view_btn.config(state="disabled")
        threading.Thread(target=self._train_worker, args=(int(self.agents.get()), self.total), daemon=True).start()
        self._poll()
        self._draw_training()

    def _train_worker(self, n_agents, epochs):
        def on_epoch(rec, total):
            self.state.update(epoch=rec["epoch"] - self.state["start"], last=rec)

        def watch(env, epoch):     # roda na thread de treino, a cada tick
            if self.watching and (epoch - self.state["start"] - 1) % VIEW_EVERY == 0:
                self.state["frame"] = dict(self.frame_of(env), epoch=epoch - self.state["start"])
                time.sleep(TRAIN_TICK_S)
            else:
                self.state["frame"] = None
        try:
            self.trainer.train(epochs, n_agents, on_epoch, watch=watch)
        finally:
            self.training = False

    def _draw_training(self):
        frame = self.state["frame"]
        if frame and self.training:
            self.draw_dynamic(frame)
            self.status.config(text=f"Visualizando a época {frame['epoch']}")
        else:
            self.canvas.delete("dyn")
        if self.training:
            self.root.after(40, self._draw_training)

    def _poll(self):
        running = self.training          # ler antes: o estado final é gravado antes de a thread encerrar
        st, last = self.state, self.state["last"]
        self.epoch_lbl.config(text=f"Época {st['epoch']} / {self.total}")
        self.progress.config(value=st["epoch"])
        if last:
            self.info_lbl.config(text=f"chegaram: {last['arrived']}/{last['n_agents']}\n"
                                      f"viagem média: {last['mean_trip_ticks']} ticks\n"
                                      f"espera média: {last['mean_wait_ticks']} ticks\n"
                                      f"duração da época: {last['makespan_ticks']} ticks")
        if running:
            self.root.after(200, self._poll)
        else:
            self.train_btn.config(state="normal")
            self.new_btn.config(state="normal")
            self.view_btn.config(state="normal" if self.trained() else "disabled")
            self.status.config(text="Execução salva em results/training.json")

    def trained(self):
        return self.trainer is not None and self.trainer.epoch > 0

    def new_run(self):
        if self.training or self.animating:
            return
        self.trainer, self.env, self.trails = None, None, {}
        self.state.update(epoch=0, last=None, start=0, frame=None)
        self.epoch_lbl.config(text=f"Época 0 / {self.total}")
        self.progress.config(value=0)
        self.info_lbl.config(text="")
        self.canvas.delete("dyn")
        self.view_btn.config(state="disabled")
        self.set_hyper_state(True)
        self.status.config(text="Nova execução: ajuste os hiperparâmetros e treine antes de visualizar.")

    def pick_spawn(self, event):
        if not self.animating:
            self.toggle_agent(event.y // CELL, event.x // CELL, (event.x % CELL) / CELL, (event.y % CELL) / CELL)

    def toggle_agent(self, r, c, fx=0.5, fy=0.5):
        spawn = CITY.street_spawn(r, c, fx, fy)
        if spawn is None:
            return
        dest = "ABCD".index(self.dest.get())
        same = [a for a in self.sel if a["spawn"] == spawn]
        if same:
            self.sel.remove(same[0])
        elif len(self.sel) < MAX_VIEW and CITY.dest_cell[dest][spawn[0]] != spawn[1]:   # não nasce já no destino
            self.sel.append(dict(cell=(r, c), spawn=spawn, dest=dest))
        else:
            return
        self.canvas.delete("dyn")
        self.env, self.trails = None, {}
        self.draw_marks()

    def clear_agents(self):
        if not self.animating:
            self.sel = []
            self.canvas.delete("dyn")
            self.env, self.trails = None, {}
            self.draw_marks()

    def draw_marks(self):
        self.canvas.delete("origin")
        for i, a in enumerate(self.sel):
            r, c = a["cell"]
            x0, y0, x1, y1 = c * CELL + 1, r * CELL + 1, (c + 1) * CELL - 1, (r + 1) * CELL - 1
            d = CITY.lane_dir[a["spawn"][0]]      # marca só a metade da célula onde fica a faixa escolhida
            if d == 1: y0 = (r + .5) * CELL
            elif d == 3: y1 = (r + .5) * CELL
            elif d == 2: x1 = (c + .5) * CELL
            else: x0 = (c + .5) * CELL
            self.canvas.create_rectangle(x0, y0, x1, y1, outline=COLORS[a["dest"]], width=3, tags="origin")
            self.canvas.create_text((x0 + x1) / 2, (y0 + y1) / 2, text=str(i + 1), tags="origin",
                                    fill=COLORS[a["dest"]], font=("Helvetica", 9, "bold"))

    def visualize(self):
        if not self.sel or not self.trained():
            return
        self.env = self.trainer.agents_env([(a["spawn"][0], a["spawn"][1], a["dest"]) for a in self.sel])
        self.env._spawn()
        self.trails = {}
        self._record_trails()
        self.animating = True
        for b in (self.train_btn, self.new_btn, self.view_btn):
            b.config(state="disabled")
        self._animate()

    def _record_trails(self):
        for veh in self.env.active.values():
            self.trails.setdefault(veh.id, (COLORS[veh.dest], []))[1].append(self.car_xy(veh))

    def _animate(self):
        env = self.env
        self.trainer.step(env)
        self._record_trails()
        self.draw_dynamic(self.frame_of(env))
        self.status.config(text=f"tick {env.t}  |  na via: {len(env.active)}")
        if env.done():
            self.status.config(text=f"{len(env.finished)}/{len(self.sel)} agentes chegaram em {env.t} ticks.")
            self.animating = False
            for b in (self.train_btn, self.new_btn, self.view_btn):
                b.config(state="normal")
            return
        self.root.after(TICK_MS, self._animate)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()
