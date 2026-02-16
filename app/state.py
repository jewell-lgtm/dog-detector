import threading
import time
from collections import deque

import numpy as np

from app.detector import Detection


class AppState:
    def __init__(self):
        self._lock = threading.Lock()
        self.latest_annotated_frame: np.ndarray | None = None
        self.latest_detections: list[Detection] = []
        self.tracker_state: dict = {}
        self.event_log: deque[str] = deque(maxlen=200)
        self.timings: dict = {
            "inference_ms": 0.0,
            "render_ms": 0.0,
            "camera_fps": 0.0,
            "inference_fps": 0.0,
        }
        self.roi_points: list[tuple[int, int]] = []
        self.frame_count: int = 0
        self.inference_count: int = 0
        self.web_clients: int = 0
        self._roi_updated: bool = False
        self._trigger_enter: bool = False
        self._trigger_leave: bool = False

    def update_frame(self, frame: np.ndarray, detections: list[Detection],
                     tracker_state: dict, inference_ms: float, render_ms: float):
        with self._lock:
            self.latest_annotated_frame = frame
            self.latest_detections = detections
            self.tracker_state = tracker_state
            self.timings["inference_ms"] = inference_ms
            self.timings["render_ms"] = render_ms

    def log_event(self, msg: str):
        with self._lock:
            ts = time.strftime("%H:%M:%S")
            self.event_log.appendleft(f"{ts} {msg}")

    def get_frame_jpeg(self) -> bytes | None:
        import cv2
        with self._lock:
            if self.latest_annotated_frame is None:
                return None
            _, buf = cv2.imencode(".jpg", self.latest_annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            return buf.tobytes()

    def set_roi_from_web(self, points: list[tuple[int, int]]):
        with self._lock:
            self.roi_points = points
            self._roi_updated = True

    def clear_roi_from_web(self):
        with self._lock:
            self.roi_points = []
            self._roi_updated = True

    def consume_roi_update(self) -> list[tuple[int, int]] | None:
        with self._lock:
            if self._roi_updated:
                self._roi_updated = False
                return list(self.roi_points)
            return None

    def set_trigger(self, event: str):
        with self._lock:
            if event == "enter":
                self._trigger_enter = True
            elif event == "leave":
                self._trigger_leave = True

    def consume_triggers(self) -> tuple[bool, bool]:
        with self._lock:
            enter = self._trigger_enter
            leave = self._trigger_leave
            self._trigger_enter = False
            self._trigger_leave = False
            return enter, leave

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "tracker": dict(self.tracker_state),
                "detections": [
                    {"bbox": d.bbox, "center": d.center, "confidence": d.confidence, "in_roi": d.in_roi, "track_id": d.track_id}
                    for d in self.latest_detections
                ],
                "event_log": list(self.event_log),
                "timings": dict(self.timings),
                "frame_count": self.frame_count,
                "inference_count": self.inference_count,
                "web_clients": self.web_clients,
            }
