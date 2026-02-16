import logging
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

import uvicorn
from PySide6.QtWidgets import QApplication

from app.config import Config
from app.main_window import MainWindow
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
    # also keep WARNING+ on stderr
    stderr = logging.StreamHandler()
    stderr.setLevel(logging.WARNING)
    stderr.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root.addHandler(stderr)


def run_web_server(port: int):
    uvicorn.run(fastapi_app, host="0.0.0.0", port=port, log_level="warning")


def main():
    _setup_logging()
    config = Config.load()
    state = AppState()
    set_state(state)

    # start web server in background thread
    web_thread = threading.Thread(target=run_web_server, args=(config.web_port,), daemon=True)
    web_thread.start()

    qt_app = QApplication(sys.argv)
    window = MainWindow(config, state)
    window.show()
    state.log_event(f"Started. Web on :{config.web_port}")
    sys.exit(qt_app.exec())


if __name__ == "__main__":
    main()
