import logging
import threading
import time
from collections.abc import Callable

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MAX_RECONNECTS = 5
RECONNECT_DELAY_S = 2.0


class CameraThread:
    def __init__(self, source: str | int = 0, on_frame: Callable[[np.ndarray], None] = lambda _: None):
        self.source = source
        self._on_frame = on_frame
        self._running = False
        self._is_network = isinstance(source, str)
        self._thread: threading.Thread | None = None

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.source)
        if not self._is_network:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return cap

    def _run(self):
        cap = self._open()
        failures = 0
        while self._running:
            ret, frame = cap.read()
            if ret:
                failures = 0
                self._on_frame(frame)
            elif self._is_network and failures < MAX_RECONNECTS:
                failures += 1
                logger.warning("Stream read failed, reconnecting (%d/%d)", failures, MAX_RECONNECTS)
                cap.release()
                time.sleep(RECONNECT_DELAY_S)
                cap = self._open()
            else:
                time.sleep(0.01)
        cap.release()

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
