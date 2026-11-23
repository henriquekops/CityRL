import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .experiments import EXPERIMENTS_DIR, slug

FIGURES_DIR = Path("results/figures")
SMOOTHING = 10
COLORS = {"Q-learning": "#1f77b4", "DQN": "#d62728", "Longest queue": "#7f7f7f"}
EVALUATION_METRICS = ("light_return", "mean_wait_ticks", "mean_trip_ticks", "makespan_ticks", "arrived",
                      "agents_created")


def load_runs():
    runs = defaultdict(list)
    for path in sorted(EXPERIMENTS_DIR.rglob("light_*.json")):
        run = json.loads(path.read_text())
        runs[run["scenario"], run["method"]].append(run)
    return runs


def crossings(runs):
    return runs[0]["environment"]["crossings"]


def moving_average(values, window=SMOOTHING):
    values = np.asarray(values, float)
    if len(values) < window:
        return values
    head = np.cumsum(values[:window - 1]) / np.arange(1, window)
    return np.concatenate([head, np.convolve(values, np.ones(window) / window, mode="valid")])


def curves(runs, key):
    length = min(len(run["epochs"]) for run in runs)
    values = np.array([[np.nan if epoch.get(key) is None else epoch[key] for epoch in run["epochs"][:length]]
                       for run in runs], float)
    smoothed = np.array([moving_average(row) for row in values])
    return np.arange(1, length + 1), np.nanmean(smoothed, axis=0), np.nanstd(smoothed, axis=0)


def plot_metric(axis, runs_by_method, key, title, ylabel):
    for method, runs in runs_by_method.items():
        epochs, mean, std = curves(runs, key)
        if np.all(np.isnan(mean)):
            continue
        axis.plot(epochs, mean, color=COLORS.get(method), label=f"{method} ({len(runs)} seeds)")
        axis.fill_between(epochs, mean - std, mean + std, color=COLORS.get(method), alpha=0.2)
    axis.set(title=title, xlabel="episode", ylabel=ylabel)
    axis.grid(alpha=0.3)
    axis.legend()


def plot_scenario(scenario, runs_by_method):
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    plot_metric(axes[0, 0], runs_by_method, "light_return", "Cumulative reward per episode (traffic lights)",
                "return")
    plot_metric(axes[0, 1], runs_by_method, "mean_wait_ticks", "Mean wait at red per agent", "ticks")
    plot_metric(axes[1, 0], runs_by_method, "mean_trip_ticks", "Mean trip time", "ticks")
    plot_metric(axes[1, 1], {m: r for m, r in runs_by_method.items() if m == "DQN"}, "light_loss",
                "DQN training loss (Huber)", "loss")
    figure.suptitle(f"{scenario}: traffic lights (moving average of {SMOOTHING} episodes)")
    figure.tight_layout()
    save(figure, f"{slug(scenario)}")


def evaluation_arrays(runs):
    return {key: np.array([run["evaluation"][key] for run in runs]) for key in EVALUATION_METRICS}


def plot_scale(runs):
    methods = sorted({method for _, method in runs}, key=list(COLORS).index)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = (("light_return", "Cumulative reward (evaluation)", "return per episode"),
              ("mean_wait_ticks", "Mean wait at red (evaluation)", "ticks"),
              ("mean_trip_ticks", "Mean trip time (evaluation)", "ticks"))
    for axis, (key, title, ylabel) in zip(axes, panels):
        for method in methods:
            groups = sorted((crossings(group), group) for (_, run_method), group in runs.items()
                            if run_method == method)
            x = [count for count, _ in groups]
            values = [evaluation_arrays(group)[key] for _, group in groups]
            axis.errorbar(x, [v.mean() for v in values], yerr=[v.std() for v in values], marker="o", capsize=4,
                          color=COLORS.get(method), label=method)
        axis.set(title=title, xlabel="crossings in the map", ylabel=ylabel, xticks=sorted({c for c in x}))
        axis.grid(alpha=0.3)
        axis.legend()
    figure.suptitle("Traffic lights as the city grows: Q-learning x DQN x longest queue")
    figure.tight_layout()
    save(figure, "scale")


def save(figure, name):
    for extension in ("png", "pdf"):
        figure.savefig(FIGURES_DIR / f"{name}.{extension}", dpi=150)
    plt.close(figure)


def evaluation_table(runs):
    lines = ["scenario,crossings,method,seeds,arrived_pct,light_return,light_return_std,mean_wait_ticks,"
             "mean_trip_ticks,makespan_ticks,train_seconds"]
    for (scenario, method), group in sorted(runs.items(), key=lambda item: (crossings(item[1]), item[0][1])):
        values = evaluation_arrays(group)
        arrived = 100 * values["arrived"].sum() / values["agents_created"].sum()
        seconds = np.mean([run["total_seconds_trained"] for run in group])
        lines.append(f"{scenario},{crossings(group)},{method},{len(group)},{arrived:.1f},"
                     f"{values['light_return'].mean():.1f},{values['light_return'].std():.1f},"
                     f"{values['mean_wait_ticks'].mean():.2f},{values['mean_trip_ticks'].mean():.2f},"
                     f"{values['makespan_ticks'].mean():.0f},{seconds:.1f}")
    return "\n".join(lines) + "\n"


def main():
    runs = load_runs()
    if not runs:
        raise SystemExit(f"No results in {EXPERIMENTS_DIR}. Run cityrl.experiments first.")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for scenario in sorted({scenario for scenario, _ in runs}):
        plot_scenario(scenario, {method: group for (run_scenario, method), group in runs.items()
                                 if run_scenario == scenario})
    plot_scale(runs)
    table = evaluation_table(runs)
    (FIGURES_DIR / "evaluation.csv").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
