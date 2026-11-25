from datetime import datetime
from pathlib import Path

LOGS_DIR = Path("results/logs")


def logger(command):
    path = LOGS_DIR / f"{command}_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.log"

    def log(message):
        print(message, flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as file:
            file.write(f"{message}\n")

    return log
