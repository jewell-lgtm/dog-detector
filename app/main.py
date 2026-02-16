import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import uvicorn

from app.config import Config
from app.pipeline import Pipeline
from app.state import AppState
from app.web.server import app as fastapi_app, set_state

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "dog-detector.log"


def _setup_logging():
    LOG_DIR.mkdir(exist_ok=True)
    handler = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=3)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
    ))
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(handler)
    stderr = logging.StreamHandler()
    stderr.setLevel(logging.WARNING)
    stderr.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root.addHandler(stderr)


def main():
    _setup_logging()
    config = Config.load()
    state = AppState()
    set_state(state)

    pipeline = Pipeline(config, state)
    pipeline.start()

    state.log_event(f"Started. Web on :{config.web_port}")
    uvicorn.run(fastapi_app, host="127.0.0.1", port=config.web_port, log_level="warning")


if __name__ == "__main__":
    main()
