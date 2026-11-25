import threading
from pathlib import Path
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from dataclasses import fields

from .city import CITY, SCENARIOS, cell_of, is_street, position_of
from .models import MODELS_DIR, describe, load_model, save_model
from .simulator import CLOSED, EAST_WEST, NORTH_SOUTH, scattered_agents
from .trainer import RESULTS_FILE, Hyperparameters, Trainer

MAX_CELL_PIXELS, MIN_CELL_PIXELS, MAP_PIXELS = 60, 6, 720
WINDOW_MARGIN_PIXELS = 100
DEFAULT_EPOCHS = 100
DEFAULT_AGENTS_PER_EPOCH = 300
VISUALIZATION_TICK_MS = 300
DEFAULT_VISUALIZED_AGENTS = 10
WATCH_EVERY_N_EPOCHS = 10
WATCH_TICK_DELAY_S = 0.015

DESTINATION_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#008080", "#9a6324", "#808000",
                      "#000075", "#f032e6"]
STREET_COLOR, BLOCK_COLOR, HOLE_COLOR = "#d9d9d9", "#f3e9d2", "black"
GREEN, RED = "#2ca02c", "#d62728"


map_pixels = MAP_PIXELS


def cell_pixels():
    return max(MIN_CELL_PIXELS, min(MAX_CELL_PIXELS, map_pixels // CITY.grid_size))


def cell_center_pixels(cell):
    row, col = position_of(cell)
    return (col + 0.5) * cell_pixels(), (row + 0.5) * cell_pixels()


def snapshot(simulation):
    cars = [(*cell_center_pixels(agent.cell), agent.destination, agent.id) for agent in simulation.agents.values()]
    return dict(signal=list(simulation.signal), all_red=list(simulation.all_red_ticks), cars=cars,
                holes=list(CITY.holes))


class TrainingProgress:

    def __init__(self, first_epoch):
        self.first_epoch = first_epoch
        self.epochs_done = 0
        self.last_record = None
        self.snapshot = None
        self.model_path = None
        self.model_error = None


class MapView(tk.Canvas):

    def __init__(self, parent):
        super().__init__(parent, bg="white", highlightthickness=0)
        self.redraw()

    def redraw(self):
        self.delete("all")
        size = CITY.grid_size * cell_pixels()
        self.config(width=size, height=size)
        self._draw_static_map()

    def _square(self, row, col, margin=0, **options):
        pixels = cell_pixels()
        return self.create_rectangle(col * pixels + margin, row * pixels + margin,
                                     (col + 1) * pixels - margin, (row + 1) * pixels - margin, **options)

    def _draw_static_map(self):
        for row in range(CITY.grid_size):
            for col in range(CITY.grid_size):
                self._square(row, col, fill=STREET_COLOR if is_street(row, col) else BLOCK_COLOR, outline="white")
        self._draw_holes(CITY.holes, tag="holes")
        for index, (row, col) in enumerate(CITY.destination_houses):
            self._square(row, col, margin=2, fill=DESTINATION_COLORS[index], outline="black")
            self.create_text(*cell_center_pixels(cell_of(row, col)), text=CITY.destination_names[index],
                             fill="white", font=("Helvetica", max(8, cell_pixels() // 4), "bold"))
        self._draw_signals([CLOSED] * CITY.n_intersections, [0] * CITY.n_intersections, tag="signals")

    def _draw_holes(self, holes, tag):
        self.delete(tag)
        for row, col in holes:
            self._square(row, col, fill=HOLE_COLOR, outline=HOLE_COLOR, tags=tag)

    def _draw_signals(self, signal, all_red, tag):
        half, bar = cell_pixels() / 2, max(2, cell_pixels() / 12)
        for intersection, cell in enumerate(CITY.intersections):
            x, y = cell_center_pixels(cell)
            traffic_flows = all_red[intersection] == 0
            east_west = GREEN if traffic_flows and signal[intersection] == EAST_WEST else RED
            north_south = GREEN if traffic_flows and signal[intersection] == NORTH_SOUTH else RED
            self.create_rectangle(x - half, y - bar, x + half, y + bar, fill=east_west, outline="", tags=tag)
            self.create_rectangle(x - bar, y - half, x + bar, y + half, fill=north_south, outline="", tags=tag)
            self.create_rectangle(x - bar, y - bar, x + bar, y + bar, fill="#333", outline="", tags=tag)

    def clear_frame(self):
        self.delete("frame")

    def show_frame(self, frame):
        self.clear_frame()
        self._draw_holes(frame["holes"], tag="holes")
        self._draw_signals(frame["signal"], frame["all_red"], tag="frame")
        radius = cell_pixels() / 5
        for x, y, destination, agent_id in frame["cars"]:
            self.create_oval(x - radius, y - radius, x + radius, y + radius, fill=DESTINATION_COLORS[destination], outline="black",
                             width=2, tags="frame")
            if cell_pixels() >= 40:
                self.create_text(x, y, text=str(agent_id + 1), fill="white", font=("Helvetica", 10, "bold"),
                                 tags="frame")


class App:
    def __init__(self, root):
        root.title("CityRL")
        self.root = root
        self.trainer = None
        self.progress = TrainingProgress(first_epoch=0)
        self.total_epochs = DEFAULT_EPOCHS
        self.is_training = self.is_animating = False
        self.stop_requested = False
        self.watching_training = False
        self.animation_job = None
        self.simulation = None
        self.model_loaded = False

        self.map = MapView(root)
        self.map.grid(row=0, column=0, padx=8, pady=8, sticky="n")
        self.resize_job = None
        side = self._scrollable_panel(root)
        self._build_run_settings(side)
        self._build_hyperparameter_panel(side)
        self._build_training_controls(side)
        self._build_visualization_controls(side)
        self.status_label.config(text="Train before visualizing.")
        self.hyperparameter_vars["scenario"].trace_add("write", lambda *_: self._on_scenario_changed())
        self._fit_window()

    def _scrollable_panel(self, root):
        root.rowconfigure(0, weight=1)
        root.columnconfigure(1, weight=1)
        outer = ttk.Frame(root)
        outer.grid(row=0, column=1, sticky="nsew", padx=(0, 8), pady=8)
        canvas = tk.Canvas(outer, highlightthickness=0, borderwidth=0)
        self.panel_outer = outer
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        panel = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=panel, anchor="nw")

        def fit(_event):
            canvas.configure(scrollregion=canvas.bbox("all"), width=panel.winfo_reqwidth(),
                             height=min(panel.winfo_reqheight(), root.winfo_screenheight() - 140))

        def wheel(event):
            if event.num == 4 or event.delta > 0:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5 or event.delta < 0:
                canvas.yview_scroll(1, "units")

        panel.bind("<Configure>", fit)
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            outer.bind_all(sequence, lambda event: wheel(event) if self._is_inside(event, outer) else None)
        return panel

    def _fit_window(self):
        """Opens the window as large as the screen allows, shrinking the map when everything does not fit."""
        root = self.root
        root.update_idletasks()
        width, height = root.winfo_screenwidth() - 40, root.winfo_screenheight() - WINDOW_MARGIN_PIXELS
        self._resize_map(self._map_space(width, height))
        root.update_idletasks()
        root.geometry(f"{min(width, root.winfo_reqwidth())}x{min(height, root.winfo_reqheight())}")
        root.bind("<Configure>", self._on_window_resized)

    def _panel_width(self):
        return self.panel_outer.winfo_reqwidth() + 8

    def _map_space(self, width, height):
        width = min(width, self.root.winfo_screenwidth() - 40)
        height = min(height, self.root.winfo_screenheight() - WINDOW_MARGIN_PIXELS)
        return max(MIN_CELL_PIXELS * CITY.grid_size, min(height - 16, width - self._panel_width() - 16))

    def _on_window_resized(self, event):
        if event.widget is not self.root:
            return
        if self.resize_job:
            self.root.after_cancel(self.resize_job)
        self.resize_job = self.root.after(150, lambda: self._resize_map(
            self._map_space(self.root.winfo_width(), self.root.winfo_height())))

    def _resize_map(self, pixels):
        global map_pixels
        self.resize_job = None
        before = cell_pixels()
        map_pixels = pixels
        if cell_pixels() != before or not self.map.find_all():
            self.map.redraw()
            if self.simulation is not None:
                self.map.show_frame(snapshot(self.simulation))

    @staticmethod
    def _is_inside(event, widget):
        x, y = widget.winfo_rootx(), widget.winfo_rooty()
        return x <= event.x_root < x + widget.winfo_width() and y <= event.y_root < y + widget.winfo_height()

    def _build_run_settings(self, parent):
        row = ttk.Frame(parent)
        row.pack(anchor="w")
        self.agents_var, self.epochs_var = tk.IntVar(value=DEFAULT_AGENTS_PER_EPOCH), tk.IntVar(value=DEFAULT_EPOCHS)
        ttk.Label(row, text="Agents per epoch").grid(row=0, column=0, sticky="w")
        ttk.Label(row, text="Epochs").grid(row=0, column=1, sticky="w", padx=(12, 0))
        ttk.Spinbox(row, from_=1, to=5000, increment=50, textvariable=self.agents_var, width=8).grid(row=1, column=0)
        ttk.Spinbox(row, from_=1, to=10000, increment=50, textvariable=self.epochs_var, width=8).grid(
            row=1, column=1, padx=(12, 0))

    def _build_hyperparameter_panel(self, parent):
        self.hyperparameter_vars, self.hyperparameter_widgets = {}, []
        groups = {}
        for setting in fields(Hyperparameters):
            groups.setdefault(setting.metadata["group"], []).append(setting)
        box = ttk.LabelFrame(parent, text="Hyperparameters")
        box.pack(fill="x", pady=(8, 0))
        for column, (group, settings) in enumerate(groups.items()):
            frame = ttk.Frame(box)
            frame.grid(row=column // 2, column=column % 2, sticky="nw", padx=6, pady=4)
            ttk.Label(frame, text=group, font=("Helvetica", 11, "bold")).grid(row=0, column=0, columnspan=2, sticky="w")
            for row, setting in enumerate(settings, start=1):
                widget, variable = self._hyperparameter_widget(frame, row, setting)
                self.hyperparameter_vars[setting.name] = variable
                self.hyperparameter_widgets.append(widget)

    @staticmethod
    def _hyperparameter_widget(frame, row, setting):
        label = setting.metadata["label"]
        if "choices" in setting.metadata:
            variable = tk.StringVar(value=setting.default)
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w")
            widget = ttk.Combobox(frame, textvariable=variable, values=list(setting.metadata["choices"]),
                                  state="readonly", width=10)
            widget.grid(row=row, column=1, padx=(6, 0))
            return widget, variable
        if setting.type is bool:
            variable = tk.BooleanVar(value=setting.default)
            widget = ttk.Checkbutton(frame, text=label, variable=variable)
            widget.grid(row=row, column=0, columnspan=2, sticky="w")
            return widget, variable
        variable = tk.IntVar(value=setting.default) if setting.type is int else tk.DoubleVar(value=setting.default)
        minimum, maximum, step = setting.metadata["range"]
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w")
        widget = ttk.Spinbox(frame, from_=minimum, to=maximum, increment=step, textvariable=variable, width=6)
        widget.grid(row=row, column=1, padx=(6, 0))
        return widget, variable

    def _build_training_controls(self, parent):
        buttons = ttk.Frame(parent)
        buttons.pack(fill="x", pady=8)
        self.train_button = ttk.Button(buttons, text="Train", command=self._start_training)
        self.train_button.pack(side="left", expand=True, fill="x")
        self.new_run_button = ttk.Button(buttons, text="New run", command=self._new_run)
        self.new_run_button.pack(side="left", expand=True, fill="x")
        self.stop_button = ttk.Button(buttons, text="Stop", command=self._stop, state="disabled")
        self.stop_button.pack(side="left", expand=True, fill="x")
        self.load_button = ttk.Button(parent, text="Load model...", command=self._load_model)
        self.load_button.pack(fill="x")
        self.model_label = ttk.Label(parent, text="", foreground="#555", wraplength=260)
        self.model_label.pack(anchor="w")
        self.epoch_label = ttk.Label(parent, text=f"Epoch 0 / {self.total_epochs}", font=("Helvetica", 14, "bold"))
        self.epoch_label.pack(anchor="w")
        self.progress_bar = ttk.Progressbar(parent, maximum=self.total_epochs)
        self.progress_bar.pack(fill="x", pady=4)
        self.watch_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(parent, text=f"Watch the training (1 epoch in {WATCH_EVERY_N_EPOCHS})", variable=self.watch_var,
                        command=lambda: setattr(self, "watching_training", self.watch_var.get())).pack(anchor="w")
        self.metrics_label = ttk.Label(parent, text="", justify="left")
        self.metrics_label.pack(anchor="w")

    def _build_visualization_controls(self, parent):
        ttk.Separator(parent).pack(fill="x", pady=10)
        ttk.Label(parent, text="Visualize agents (no learning)").pack(anchor="w")
        row = ttk.Frame(parent)
        row.pack(anchor="w", pady=(4, 0))
        ttk.Label(row, text="Agents").pack(side="left")
        self.spawn_var = tk.IntVar(value=DEFAULT_VISUALIZED_AGENTS)
        self.spawn_box = ttk.Spinbox(row, from_=0, to=CITY.max_spawned_agents, increment=10,
                                     textvariable=self.spawn_var, width=6)
        self.spawn_box.pack(side="left", padx=4)
        self.spawn_limit_label = ttk.Label(parent, text="", foreground="#555", wraplength=400)
        self.spawn_limit_label.pack(anchor="w")
        self._update_spawn_limit()
        self.visualize_button = ttk.Button(parent, text="Visualize", command=self._start_visualization, state="disabled")
        self.visualize_button.pack(fill="x", pady=(8, 2))
        self.status_label = ttk.Label(parent, text="", foreground="#555", wraplength=400)
        self.status_label.pack(anchor="w")

    def _set_buttons(self, train, new_run, visualize, stop=False):
        for button, enabled in ((self.train_button, train), (self.new_run_button, new_run),
                                (self.visualize_button, visualize), (self.stop_button, stop),
                                (self.load_button, train)):
            button.config(state="normal" if enabled else "disabled")

    def _is_trained(self):
        return self.trainer is not None and (self.trainer.epochs_done > 0 or self.model_loaded)

    def _read_hyperparameters(self):
        try:
            return Hyperparameters(**{field: var.get() for field, var in self.hyperparameter_vars.items()})
        except tk.TclError:
            messagebox.showerror("Hyperparameters", "A hyperparameter has an invalid value.")
            return None

    def _set_hyperparameters_enabled(self, enabled):
        for widget in self.hyperparameter_widgets:
            enabled_state = "readonly" if isinstance(widget, ttk.Combobox) else "normal"
            widget.config(state=enabled_state if enabled else "disabled")

    def _reset_visualization(self):
        self.map.clear_frame()
        self.simulation = None

    def _start_training(self):
        if self.is_training:
            return
        if self.trainer is None:
            hyperparameters = self._read_hyperparameters()
            if hyperparameters is None:
                return
            self.trainer = Trainer(hyperparameters)
            self._on_city_changed()
            self._set_hyperparameters_enabled(False)
        self.is_training, self.stop_requested = True, False
        self.total_epochs = max(1, int(self.epochs_var.get()))
        self.progress = TrainingProgress(first_epoch=self.trainer.epochs_done)
        self.progress_bar.config(maximum=self.total_epochs)
        self.epoch_label.config(text=f"Epoch 0 / {self.total_epochs}")
        self._reset_visualization()
        self._set_buttons(train=False, new_run=False, visualize=False, stop=True)
        threading.Thread(target=self._train_in_background, args=(int(self.agents_var.get()), self.total_epochs),
                         daemon=True).start()
        self._poll_training()
        self._draw_watched_epoch()

    def _train_in_background(self, n_agents, epochs):
        progress = self.progress

        def on_epoch(record):
            progress.epochs_done = record["epoch"] - progress.first_epoch
            progress.last_record = record

        try:
            self.trainer.train(epochs, n_agents, on_epoch, should_stop=lambda: self.stop_requested,
                               on_tick=self._on_training_tick)
            if progress.epochs_done > 0:
                try:
                    progress.model_path = save_model(self.trainer)
                except Exception as error:
                    progress.model_error = str(error)
        finally:
            self.is_training = False

    def _on_training_tick(self, simulation, epoch_number):
        epoch_in_run = epoch_number - self.progress.first_epoch
        if self.watching_training and (epoch_in_run - 1) % WATCH_EVERY_N_EPOCHS == 0:
            self.progress.snapshot = dict(snapshot(simulation), epoch=epoch_in_run)
            time.sleep(WATCH_TICK_DELAY_S)
        else:
            self.progress.snapshot = None

    def _draw_watched_epoch(self):
        watched = self.progress.snapshot
        if watched and self.is_training:
            self.map.show_frame(watched)
            self.status_label.config(text=f"Watching epoch {watched['epoch']}")
        else:
            self.map.clear_frame()
        if self.is_training:
            self.root.after(40, self._draw_watched_epoch)

    def _poll_training(self):
        still_training = self.is_training
        progress, record = self.progress, self.progress.last_record
        self.epoch_label.config(text=f"Epoch {progress.epochs_done} / {self.total_epochs}")
        self.progress_bar.config(value=progress.epochs_done)
        if record:
            self.metrics_label.config(text=f"arrived: {record['arrived']}/{record['agents_created']}\n"
                                           f"mean trip: {record['mean_trip_ticks']} ticks\n"
                                           f"mean wait: {record['mean_wait_ticks']} ticks\n"
                                           f"hole hits: {record['hole_hits']}\n"
                                           f"light return: {record['light_return']}\n"
                                           f"agents return: {record['router_return']}\n"
                                           f"epoch length: {record['makespan_ticks']} ticks\n"
                                           f"epoch time: {record['seconds']} s")
        if still_training:
            self.root.after(200, self._poll_training)
        else:
            self._set_buttons(train=True, new_run=True, visualize=self._is_trained())
            if not self.stop_requested:
                message = f"Run saved to {RESULTS_FILE}"
            elif progress.epochs_done == 0:
                message = "Stopped before the first epoch finished: nothing saved."
            else:
                message = f"Stopped after {progress.epochs_done} epochs. Run saved to {RESULTS_FILE}"
            if progress.model_path:
                message += f"\nModel saved to {progress.model_path}"
                self.model_label.config(text=f"Model: {describe(self.trainer)}")
            elif progress.model_error:
                message += f"\nModel NOT saved: {progress.model_error}"
            self.status_label.config(text=message)

    def _on_scenario_changed(self):
        if self.trainer is not None:
            return
        scenario = self.hyperparameter_vars["scenario"].get()
        CITY.configure(scenario)
        self.agents_var.set(SCENARIOS[scenario].default_agents)
        self._on_city_changed()

    def _update_spawn_limit(self):
        limit = CITY.max_spawned_agents
        self.spawn_box.config(to=limit)
        if self._spawn_count() > limit:
            self.spawn_var.set(limit)
        self.spawn_limit_label.config(text=f"max {limit}: street cells that are not a crossing, hole or "
                                           f"destination door (one agent per cell, random destinations)")

    def _spawn_count(self):
        try:
            return max(0, int(self.spawn_var.get()))
        except (tk.TclError, ValueError):
            return 0

    def _on_city_changed(self):
        if self.root.winfo_ismapped():
            self.map.delete("all")
            self._resize_map(self._map_space(self.root.winfo_width(), self.root.winfo_height()))
        else:
            self.map.redraw()
        self._update_spawn_limit()

    def _new_run(self):
        if self.is_training or self.is_animating:
            return
        self.trainer = None
        self.model_loaded = False
        self.model_label.config(text="")
        self.progress = TrainingProgress(first_epoch=0)
        self._reset_visualization()
        self.epoch_label.config(text=f"Epoch 0 / {self.total_epochs}")
        self.progress_bar.config(value=0)
        self.metrics_label.config(text="")
        self._set_buttons(train=True, new_run=True, visualize=False)
        self._set_hyperparameters_enabled(True)
        self.status_label.config(text="New run: adjust the hyperparameters and train before visualizing.")

    def _stop(self):
        if self.is_training:
            self.stop_requested = True
            self.status_label.config(text="Stopping...")
            self.stop_button.config(state="disabled")
        elif self.is_animating:
            self.root.after_cancel(self.animation_job)
            self.is_animating = False
            self.status_label.config(text=f"Stopped at tick {self.simulation.time}.")
            self._set_buttons(train=True, new_run=True, visualize=True)

    def _start_visualization(self):
        if not self._is_trained():
            return
        agents = scattered_agents(min(self._spawn_count(), CITY.max_spawned_agents))
        if not agents:
            self.status_label.config(text="Choose how many agents to spawn.")
            return
        self.visualized_agents = len(agents)
        self.simulation = self.trainer.simulation_for_agents(agents, max_agents_on_map=len(agents))
        self.simulation.spawn_due_agents()
        self.is_animating = True
        self._set_buttons(train=False, new_run=False, visualize=False, stop=True)
        self._animate()

    def _animate(self):
        simulation = self.simulation
        self.trainer.step(simulation)
        self.map.show_frame(snapshot(simulation))
        self.status_label.config(text=f"tick {simulation.time}  |  on the map: {len(simulation.agents)}")
        if simulation.is_done():
            self.status_label.config(
                text=f"{len(simulation.trips)}/{self.visualized_agents} agents arrived in {simulation.time} ticks.")
            self.is_animating = False
            self._set_buttons(train=True, new_run=True, visualize=True)
        else:
            self.animation_job = self.root.after(VISUALIZATION_TICK_MS, self._animate)

    def _load_model(self):
        if self.is_training or self.is_animating:
            return
        folder = MODELS_DIR if MODELS_DIR.exists() else Path(".")
        path = filedialog.askopenfilename(title="Load model", initialdir=folder.resolve(),
                                          filetypes=[("CityRL model", "*.pt"), ("All files", "*")])
        if not path:
            return
        try:
            trainer = load_model(path)
        except Exception as error:
            messagebox.showerror("Load model", f"Could not load {path}:\n{error}")
            return
        self.trainer, self.model_loaded = trainer, True
        for name, variable in self.hyperparameter_vars.items():
            variable.set(getattr(trainer.hyperparameters, name))
        self._set_hyperparameters_enabled(False)
        self.agents_var.set(CITY.scenario.default_agents)
        self.progress = TrainingProgress(first_epoch=trainer.epochs_done)
        self.epoch_label.config(text=f"Epoch 0 / {self.total_epochs}")
        self.progress_bar.config(value=0)
        self.metrics_label.config(text="")
        self._reset_visualization()
        self._on_city_changed()
        self._set_buttons(train=True, new_run=True, visualize=True)
        self.model_label.config(text=f"Model: {describe(trainer)}")
        self.status_label.config(text="Model loaded: choose how many agents and press Visualize (no learning, "
                                      "greedy policy). Train continues training it.")


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()
