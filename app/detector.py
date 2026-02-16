from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

DOG_CLASS_ID = 16


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    center: tuple[int, int]
    confidence: float
    track_id: int | None = None
    in_roi: bool = False


class DogDetector:
    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.4):
        self.model = YOLO(model_name)
        self.confidence = confidence

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.track(frame, conf=self.confidence, persist=True, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) != DOG_CLASS_ID:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                tid = int(box.id[0]) if box.id is not None else None
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    center=(cx, cy),
                    confidence=float(box.conf[0]),
                    track_id=tid,
                ))
        return detections
