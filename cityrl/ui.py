"""Tkinter interface: train the agents and visualize chosen agents running the learned policy."""
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from dataclasses import fields
from typing import NamedTuple

from .city import CITY, DESTINATION_HOUSES, DESTINATION_NAMES, GRID_SIZE, cell_of, is_street, position_of
from .simulator import CLOSED, EAST_WEST, NORTH_SOUTH
from .trainer import Hyperparameters, Trainer

CELL_PIXELS = 60
DEFAULT_EPOCHS = 100
DEFAULT_AGENTS_PER_EPOCH = 20
MAX_VISUALIZED_AGENTS = 10
VISUALIZATION_TICK_MS = 300         # slow motion when following the chosen agents
WATCH_EVERY_N_EPOCHS = 10           # when watching the training, show one epoch out of this many
WATCH_TICK_DELAY_S = 0.015          # pause per tick in the epochs that are shown

DESTINATION_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231"]
STREET_COLOR, BLOCK_COLOR, HOLE_COLOR = "#d9d9d9", "#f3e9d2", "black"
GREEN, RED = "#2ca02c", "#d62728"


def cell_center_pixels(cell):
    row, col = position_of(cell)
    return (col + 0.5) * CELL_PIXELS, (row + 0.5) * CELL_PIXELS


def snapshot(simulation):
    """What the map needs to draw one moment: signals and agents (x, y, destination, id)."""
    cars = [(*cell_center_pixels(agent.cell), agent.destination, agent.id) for agent in simulation.agents.values()]
    return dict(signal=list(simulation.signal), all_red=list(simulation.all_red_ticks), cars=cars)


class SelectedAgent(NamedTuple):
    cell: int
    destination: int


class TrainingProgress:
    """Written by the training thread, read by the interface."""

    def __init__(self, first_epoch):
        self.first_epoch = first_epoch      # epochs trained before this run started
        self.epochs_done = 0                # epochs finished in this run
        self.last_record = None
        self.snapshot = None                # moment of the epoch being watched, if any


# ---------- map drawing ----------
class MapView(tk.Canvas):
    """The city: static map (streets, hole, destinations, traffic lights) plus a layer redrawn on every frame."""

    def __init__(self, parent, on_click):
        size = GRID_SIZE * CELL_PIXELS
        super().__init__(parent, width=size, height=size, bg="white", highlightthickness=0)
        self.bind("<Button-1>", lambda event: on_click(event.y // CELL_PIXELS, event.x // CELL_PIXELS))
        self._draw_static_map()

    def _square(self, row, col, margin=0, **options):
        return self.create_rectangle(col * CELL_PIXELS + margin, row * CELL_PIXELS + margin,
                                     (col + 1) * CELL_PIXELS - margin, (row + 1) * CELL_PIXELS - margin, **options)

    def _draw_static_map(self):
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                self._square(row, col, fill=STREET_COLOR if is_street(row, col) else BLOCK_COLOR, outline="white")
        self._square(*CITY.hole, fill=HOLE_COLOR, outline=HOLE_COLOR)
        for index, (row, col) in enumerate(DESTINATION_HOUSES):
            self._square(row, col, margin=2, fill=DESTINATION_COLORS[index], outline="black")
            self.create_text((col + 0.5) * CELL_PIXELS, (row + 0.5) * CELL_PIXELS, text=DESTINATION_NAMES[index],
                             fill="white", font=("Helvetica", 14, "bold"))
        self._draw_signals([CLOSED] * CITY.n_intersections, [0] * CITY.n_intersections, tag="signals")

    def _draw_signals(self, signal, all_red, tag):
        """One bar per axis at every intersection: green where traffic flows, red otherwise."""
        half = CELL_PIXELS / 2
        for intersection, cell in enumerate(CITY.intersections):
            x, y = cell_center_pixels(cell)
            traffic_flows = all_red[intersection] == 0
            east_west = GREEN if traffic_flows and signal[intersection] == EAST_WEST else RED
            north_south = GREEN if traffic_flows and signal[intersection] == NORTH_SOUTH else RED
            self.create_rectangle(x - half, y - 5, x + half, y + 5, fill=east_west, outline="", tags=tag)
            self.create_rectangle(x - 5, y - half, x + 5, y + half, fill=north_south, outline="", tags=tag)
            self.create_rectangle(x - 5, y - 5, x + 5, y + 5, fill="#333", outline="", tags=tag)

    def clear_frame(self):
        self.delete("frame")

    def show_frame(self, frame):
        """Draw signals and agents of one moment (replaces the previous frame)."""
        self.clear_frame()
        self._draw_signals(frame["signal"], frame["all_red"], tag="frame")
        for x, y, destination, agent_id in frame["cars"]:
            self.create_oval(x - 12, y - 12, x + 12, y + 12, fill=DESTINATION_COLORS[destination], outline="black",
                             width=2, tags="frame")
            self.create_text(x, y, text=str(agent_id + 1), fill="white", font=("Helvetica", 10, "bold"), tags="frame")

    def show_selection(self, agents):
        """Mark the chosen starting cells with a numbered square in the color of the agent's destination."""
        self.delete("selection")
        for number, agent in enumerate(agents, start=1):
            row, col = position_of(agent.cell)
            color = DESTINATION_COLORS[agent.destination]
            self._square(row, col, margin=2, outline=color, width=3, tags="selection")
            self.create_text(*cell_center_pixels(agent.cell), text=str(number), fill=color, tags="selection",
                             font=("Helvetica", 11, "bold"))


# ---------- application ----------
class App:
    def __init__(self, root):
        root.title("CityRL")
        self.root = root
        self.trainer = None                     # created at the first training, with the hyperparameters on screen
        self.progress = TrainingProgress(first_epoch=0)
        self.total_epochs = DEFAULT_EPOCHS
        self.is_training = self.is_animating = False
        self.stop_requested = False             # read by the training thread
        self.watching_training = False          # read by the training thread
        self.animation_job = None               # pending `after` call of the animation
        self.selection = []                     # SelectedAgent list, one per cell
        self.simulation = None                  # the simulation being visualized

        self.map = MapView(root, on_click=self._on_map_click)
        self.map.grid(row=0, column=0, padx=8, pady=8)
        side = ttk.Frame(root)
        side.grid(row=0, column=1, sticky="n", padx=(0, 8), pady=8)
        self._build_run_settings(side)
        self._build_hyperparameter_panel(side)
        self._build_training_controls(side)
        self._build_visualization_controls(side)
        self.status_label.config(text="Train before visualizing.")
        self._toggle_agent(0, 7)                # initial agent: top street, destination A

    # ---------- building the panel ----------
    def _build_run_settings(self, parent):
        row = ttk.Frame(parent)
        row.pack(anchor="w")
        self.agents_var, self.epochs_var = tk.IntVar(value=DEFAULT_AGENTS_PER_EPOCH), tk.IntVar(value=DEFAULT_EPOCHS)
        ttk.Label(row, text="Agents per epoch").grid(row=0, column=0, sticky="w")
        ttk.Label(row, text="Epochs").grid(row=0, column=1, sticky="w", padx=(12, 0))
        ttk.Spinbox(row, from_=1, to=500, increment=5, textvariable=self.agents_var, width=8).grid(row=1, column=0)
        ttk.Spinbox(row, from_=1, to=10000, increment=50, textvariable=self.epochs_var, width=8).grid(
            row=1, column=1, padx=(12, 0))

    def _build_hyperparameter_panel(self, parent):
        """One widget per field of `Hyperparameters`, grouped in columns by the group declared in the field."""
        self.hyperparameter_vars, self.hyperparameter_widgets = {}, []
        groups = {}
        for setting in fields(Hyperparameters):
            groups.setdefault(setting.metadata["group"], []).append(setting)
        box = ttk.LabelFrame(parent, text="Hyperparameters")
        box.pack(fill="x", pady=(8, 0))
        for column, (group, settings) in enumerate(groups.items()):
            frame = ttk.Frame(box)
            frame.grid(row=0, column=column, sticky="n", padx=6, pady=4)
            ttk.Label(frame, text=group, font=("Helvetica", 11, "bold")).grid(row=0, column=0, columnspan=2, sticky="w")
            for row, setting in enumerate(settings, start=1):
                widget, variable = self._hyperparameter_widget(frame, row, setting)
                self.hyperparameter_vars[setting.name] = variable
                self.hyperparameter_widgets.append(widget)

    @staticmethod
    def _hyperparameter_widget(frame, row, setting):
        label = setting.metadata["label"]
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
        ttk.Label(parent, text=f"Visualize agents (up to {MAX_VISUALIZED_AGENTS})").pack(anchor="w")
        row = ttk.Frame(parent)
        row.pack(anchor="w")
        ttk.Label(row, text="Destination").pack(side="left")
        self.destination_var = tk.StringVar(value=DESTINATION_NAMES[0])
        ttk.Combobox(row, textvariable=self.destination_var, values=list(DESTINATION_NAMES), state="readonly",
                     width=3).pack(side="left", padx=4)
        self.visualize_button = ttk.Button(parent, text="Visualize", command=self._start_visualization, state="disabled")
        self.visualize_button.pack(fill="x", pady=(8, 2))
        ttk.Button(parent, text="Clear agents", command=self._clear_agents).pack(fill="x")
        self.status_label = ttk.Label(parent, text="", foreground="#555", wraplength=200)
        self.status_label.pack(anchor="w")

    # ---------- small helpers ----------
    def _set_buttons(self, train, new_run, visualize, stop=False):
        for button, enabled in ((self.train_button, train), (self.new_run_button, new_run),
                                (self.visualize_button, visualize), (self.stop_button, stop)):
            button.config(state="normal" if enabled else "disabled")

    def _is_trained(self):
        return self.trainer is not None and self.trainer.epochs_done > 0

    def _read_hyperparameters(self):
        try:
            return Hyperparameters(**{field: var.get() for field, var in self.hyperparameter_vars.items()})
        except tk.TclError:
            messagebox.showerror("Hyperparameters", "A hyperparameter has an invalid value.")
            return None

    def _set_hyperparameters_enabled(self, enabled):
        for widget in self.hyperparameter_widgets:
            widget.config(state="normal" if enabled else "disabled")

    def _reset_visualization(self):
        self.map.clear_frame()
        self.simulation = None

    # ---------- training ----------
    def _start_training(self):
        if self.is_training:
            return
        if self.trainer is None:                # first training of a run: fix the hyperparameters
            hyperparameters = self._read_hyperparameters()
            if hyperparameters is None:
                return
            self.trainer = Trainer(hyperparameters)
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
        finally:
            self.is_training = False

    def _on_training_tick(self, simulation, epoch_number):
        """Runs in the training thread after every tick: keep a snapshot of the epochs that are being watched."""
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
        still_training = self.is_training       # read first: the last epoch is stored before the thread ends
        progress, record = self.progress, self.progress.last_record
        self.epoch_label.config(text=f"Epoch {progress.epochs_done} / {self.total_epochs}")
        self.progress_bar.config(value=progress.epochs_done)
        if record:
            self.metrics_label.config(text=f"arrived: {record['arrived']}/{record['agents_created']}\n"
                                           f"mean trip: {record['mean_trip_ticks']} ticks\n"
                                           f"mean wait: {record['mean_wait_ticks']} ticks\n"
                                           f"hole hits: {record['hole_hits']}\n"
                                           f"epoch length: {record['makespan_ticks']} ticks\n"
                                           f"epoch time: {record['seconds']} s")
        if still_training:
            self.root.after(200, self._poll_training)
        else:
            self._set_buttons(train=True, new_run=True, visualize=self._is_trained())
            if not self.stop_requested:
                message = "Run saved to results/training.json"
            elif progress.epochs_done == 0:
                message = "Stopped before the first epoch finished: nothing saved."
            else:
                message = f"Stopped after {progress.epochs_done} epochs. Run saved to results/training.json"
            self.status_label.config(text=message)

    def _new_run(self):
        """Discard the learned policy and unlock the hyperparameters: the next training is a new run."""
        if self.is_training or self.is_animating:
            return
        self.trainer = None
        self.progress = TrainingProgress(first_epoch=0)
        self._reset_visualization()
        self.epoch_label.config(text=f"Epoch 0 / {self.total_epochs}")
        self.progress_bar.config(value=0)
        self.metrics_label.config(text="")
        self._set_buttons(train=True, new_run=True, visualize=False)
        self._set_hyperparameters_enabled(True)
        self.status_label.config(text="New run: adjust the hyperparameters and train before visualizing.")

    def _stop(self):
        """Stop whatever is running: the training (after discarding the unfinished epoch) or the animation."""
        if self.is_training:
            self.stop_requested = True
            self.status_label.config(text="Stopping...")
            self.stop_button.config(state="disabled")
        elif self.is_animating:
            self.root.after_cancel(self.animation_job)
            self.is_animating = False
            self.status_label.config(text=f"Stopped at tick {self.simulation.time}.")
            self._set_buttons(train=True, new_run=True, visualize=True)

    # ---------- choosing agents ----------
    def _on_map_click(self, row, col):
        if not self.is_animating:
            self._toggle_agent(row, col)

    def _toggle_agent(self, row, col):
        """Click on a street cell: add an agent there (with the chosen destination) or remove the one already there."""
        cell = cell_of(row, col)
        if not CITY.can_place_agent(cell):
            return
        destination = DESTINATION_NAMES.index(self.destination_var.get())
        already_there = [agent for agent in self.selection if agent.cell == cell]
        if already_there:
            self.selection.remove(already_there[0])
        elif len(self.selection) < MAX_VISUALIZED_AGENTS and CITY.door_cell[destination] != cell:
            self.selection.append(SelectedAgent(cell, destination))
        else:
            return
        self._reset_visualization()
        self.map.show_selection(self.selection)

    def _clear_agents(self):
        if not self.is_animating:
            self.selection = []
            self._reset_visualization()
            self.map.show_selection(self.selection)

    # ---------- visualization ----------
    def _start_visualization(self):
        if not self.selection or not self._is_trained():
            return
        agents = [(agent.cell, agent.destination) for agent in self.selection]
        self.simulation = self.trainer.simulation_for_agents(agents)
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
                text=f"{len(simulation.trips)}/{len(self.selection)} agents arrived in {simulation.time} ticks.")
            self.is_animating = False
            self._set_buttons(train=True, new_run=True, visualize=True)
        else:
            self.animation_job = self.root.after(VISUALIZATION_TICK_MS, self._animate)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()
