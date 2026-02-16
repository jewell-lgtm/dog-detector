import cv2
import numpy as np

from app.detector import Detection
from app.roi import ROI


class OverlayPainter:
    def draw(self, frame: np.ndarray, detections: list[Detection], roi: ROI, dog_inside: bool) -> np.ndarray:
        out = frame.copy()
        self._draw_roi(out, roi, dog_inside)
        self._draw_detections(out, detections)
        return out

    def _draw_roi(self, frame: np.ndarray, roi: ROI, dog_inside: bool):
        pts = roi.polygon_array()
        if pts is None:
            return
        color = (0, 0, 255) if dog_inside else (0, 255, 0)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

    def _draw_detections(self, frame: np.ndarray, detections: list[Detection]):
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            color = (0, 0, 255) if d.in_roi else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            tid = f"#{d.track_id}" if d.track_id is not None else "?"
            label = f"{tid} {d.confidence:.0%}"
            cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.circle(frame, d.center, 4, color, -1)
