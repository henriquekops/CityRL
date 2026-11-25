import argparse
import json
import re
import time
from dataclasses import fields, replace
from pathlib import Path

import numpy as np

from .city import SCENARIOS
from .logs import logger
from .models import model_path, router_path, save_model
from .trainer import METHODS, Hyperparameters, Trainer

EXPERIMENTS_DIR = Path("results/experiments")
DEFAULT_PRETRAIN_EPOCHS = 100


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def result_path(scenario, method, seed):
    return EXPERIMENTS_DIR / slug(scenario) / f"light_{slug(method)}_seed{seed}.json"


def pretrained_router(trainer, scenario, seed, n_agents):
    path = router_path(scenario, seed)
    if path.exists():
        trainer.router.q_values = np.load(path)
        trainer.router_frozen = True
    else:
        trainer.pretrain_router(n_agents)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, trainer.router.q_values)


def run(scenario, method, seed, epochs, n_agents, evaluation_episodes, overrides=None, hole_seed=0,
        pretrain_epochs=DEFAULT_PRETRAIN_EPOCHS):
    settings = dict(overrides or {}, scenario=scenario, light_method=method, router_pretrain_epochs=pretrain_epochs)
    trainer = Trainer(replace(Hyperparameters(), **settings), seed=seed, hole_seed=hole_seed)
    started = time.perf_counter()
    if pretrain_epochs:
        pretrained_router(trainer, scenario, seed, n_agents)
    trainer.train(epochs, n_agents)
    evaluation = trainer.evaluate(evaluation_episodes, n_agents)
    summary = dict(trainer._run_summary(), scenario=scenario, agent="light", method=method, evaluation=evaluation,
                   wall_seconds=round(time.perf_counter() - started, 1))
    path = result_path(scenario, method, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    save_model(trainer, model_path(scenario, method, seed))
    return summary


def parse_overrides(pairs):
    types = {setting.name: setting.type for setting in fields(Hyperparameters)}
    overrides = {}
    for pair in pairs:
        name, value = pair.split("=", 1)
        kind = types[name]
        overrides[name] = value.lower() in ("1", "true", "yes") if kind is bool else kind(value)
    return overrides


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS), default=list(SCENARIOS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seeds", type=int, default=5, help="number of seeds")
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--agents-per-epoch", type=int, help="default: each scenario's own")
    parser.add_argument("--pretrain-epochs", type=int, default=DEFAULT_PRETRAIN_EPOCHS,
                        help="router pretraining before it is frozen (0: router learns together with the lights)")
    parser.add_argument("--evaluation-episodes", type=int, default=20)
    parser.add_argument("--hole-seed", type=int, default=0, help="same holes for every run being compared")
    parser.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE", help="override a hyperparameter")
    parser.add_argument("--overwrite", action="store_true", help="run again what already has results")

    args = parser.parse_args()
    overrides = parse_overrides(args.set)
    log = logger("experiments")

    for scenario in args.scenarios:
        n_agents = args.agents_per_epoch or SCENARIOS[scenario].default_agents

        for seed in range(args.first_seed, args.first_seed + args.seeds):

            for method in args.methods:

                if result_path(scenario, method, seed).exists() and not args.overwrite:
                    log(f"{scenario} | {method} seed {seed}: already done, skipped")
                    continue

                summary = run(scenario, method, seed, args.epochs, n_agents, args.evaluation_episodes, overrides,
                              args.hole_seed, args.pretrain_epochs)
                evaluation = summary["evaluation"]
                log(f"{scenario} | {method:13} seed {seed}: "
                    f"arrived {evaluation['arrived']:.0f}/{evaluation['agents_created']:.0f}, "
                    f"trip {evaluation['mean_trip_ticks']:.2f}, wait {evaluation['mean_wait_ticks']:.2f} ticks, "
                    f"light return {evaluation['light_return']:.1f}, {summary['wall_seconds']} s")


if __name__ == "__main__":
    main()
