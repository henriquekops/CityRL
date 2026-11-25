import argparse
import re
import time
from dataclasses import asdict, fields, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from .city import CITY, SCENARIOS
from .logs import logger
from .trainer import LONGEST_QUEUE, METHODS, Hyperparameters, Trainer
from .traffic_light import TrafficLightAgent
from .traffic_light_dqn import TrafficLightDQN

MODELS_DIR = Path("results/models")
FORMAT_VERSION = 1


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def model_path(scenario, method, seed):
    return MODELS_DIR / slug(scenario) / f"model_{slug(method)}_seed{seed}.pt"


def router_path(scenario, seed):
    return MODELS_DIR / slug(scenario) / f"router_seed{seed}.npy"


def default_model_path(trainer):
    hp = trainer.hyperparameters
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return MODELS_DIR / slug(hp.scenario) / f"ui_{slug(hp.light_method)}_seed{trainer.seed}_{stamp}.pt"


def save_model(trainer, path=None):
    path = Path(path or default_model_path(trainer))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(
        format=FORMAT_VERSION, seed=trainer.seed, hyperparameters=asdict(trainer.hyperparameters),
        scenario=CITY.scenario.name, holes=[list(hole) for hole in CITY.holes],
        epochs_trained=trainer.epochs_done, router_frozen=trainer.router_frozen,
        router=torch.from_numpy(np.array(trainer.router.q_values)),
        light=_light_state(trainer.light_agent), history=trainer.history), path)
    return path


def _light_state(light_agent):
    if isinstance(light_agent, TrafficLightDQN):
        dqn = light_agent.dqn
        return dict(kind="dqn", online=dqn.online.state_dict(), target=dqn.target.state_dict(),
                    optimizer=dqn.optimizer.state_dict(), transitions=dqn.transitions, updates=dqn.updates)
    if isinstance(light_agent, TrafficLightAgent):
        return dict(kind="q_table", states=[list(state) for state in light_agent.q_table],
                    values=[[float(value) for value in values] for values in light_agent.q_table.values()])
    return dict(kind="none")


def load_model(path):
    saved = torch.load(Path(path), map_location="cpu", weights_only=True)
    known = {setting.name for setting in fields(Hyperparameters)}
    hyperparameters = replace(Hyperparameters(), **{k: v for k, v in saved["hyperparameters"].items() if k in known})
    trainer = Trainer(hyperparameters, seed=saved["seed"], hole_seed=0)
    CITY.set_holes(saved["holes"])
    router = saved["router"].numpy()
    if router.shape != trainer.router.q_values.shape:
        raise ValueError(f"router table {router.shape} does not fit the map {trainer.router.q_values.shape}")
    trainer.router.q_values = router.copy()
    trainer.router_frozen = True
    _load_light_state(trainer.light_agent, saved["light"])
    trainer.history = list(saved.get("history") or [])
    trainer.loaded_epochs = saved["epochs_trained"]
    return trainer


def _load_light_state(light_agent, state):
    if isinstance(light_agent, TrafficLightDQN):
        dqn = light_agent.dqn
        dqn.online.load_state_dict(state["online"])
        dqn.target.load_state_dict(state["target"])
        dqn.optimizer.load_state_dict(state["optimizer"])
        dqn.transitions, dqn.updates = state["transitions"], state["updates"]
    elif isinstance(light_agent, TrafficLightAgent):
        for state_key, values in zip(state["states"], state["values"]):
            light_agent.q_table[tuple(int(item) for item in state_key)] = np.array(values)


def describe(trainer):
    hp = trainer.hyperparameters
    return f"{hp.light_method} | {hp.scenario} | {getattr(trainer, 'loaded_epochs', trainer.epochs_done)} epochs"


def generate(scenario, method, seed, epochs, n_agents, overrides, hole_seed, pretrain_epochs, evaluation_episodes,
             log=print):
    from .experiments import pretrained_router
    settings = dict(overrides, scenario=scenario, light_method=method, router_pretrain_epochs=pretrain_epochs)
    trainer = Trainer(replace(Hyperparameters(), **settings), seed=seed, hole_seed=hole_seed)
    started = time.perf_counter()
    if pretrain_epochs:
        pretrained_router(trainer, scenario, seed, n_agents)
    trainer.train(epochs if method != LONGEST_QUEUE else 0, n_agents, save=False)
    path = save_model(trainer, model_path(scenario, method, seed))
    message = f"{scenario} | {method:13} seed {seed}: {path}, {time.perf_counter() - started:.0f} s"
    if evaluation_episodes:
        evaluation = trainer.evaluate(evaluation_episodes, n_agents)
        message += (f" | eval: arrived {evaluation['arrived']:.0f}/{evaluation['agents_created']:.0f}, "
                    f"trip {evaluation['mean_trip_ticks']:.2f} ticks")
    log(message)


def main():
    from .experiments import DEFAULT_PRETRAIN_EPOCHS, parse_overrides
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS), default=list(SCENARIOS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--agents-per-epoch", type=int, help="default: each scenario's own")
    parser.add_argument("--pretrain-epochs", type=int, default=DEFAULT_PRETRAIN_EPOCHS)
    parser.add_argument("--hole-seed", type=int, default=0, help="same holes as cityrl.experiments")
    parser.add_argument("--evaluation-episodes", type=int, default=0, help="also evaluate each model (default: no)")
    parser.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE", help="override a hyperparameter")
    parser.add_argument("--overwrite", action="store_true", help="train again models that already exist")
    args = parser.parse_args()
    overrides = parse_overrides(args.set)
    log = logger("models")
    for scenario in args.scenarios:
        n_agents = args.agents_per_epoch or SCENARIOS[scenario].default_agents
        for method in args.methods:
            if model_path(scenario, method, args.seed).exists() and not args.overwrite:
                log(f"{scenario} | {method} seed {args.seed}: model already exists, skipped")
                continue
            generate(scenario, method, args.seed, args.epochs, n_agents, overrides, args.hole_seed,
                     args.pretrain_epochs, args.evaluation_episodes, log)


if __name__ == "__main__":
    main()
