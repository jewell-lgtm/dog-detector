import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal


class CameraThread(QThread):
    frame_ready = Signal(np.ndarray)

    def __init__(self, device: int = 0, parent=None):
        super().__init__(parent)
        self.device = device
        self._running = True

    def run(self):
        cap = cv2.VideoCapture(self.device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        while self._running:
            ret, frame = cap.read()
            if ret:
                self.frame_ready.emit(frame)
            else:
                self.msleep(10)
        cap.release()

    def stop(self):
        self._running = False
        self.wait()
