import time

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.state import AppState


class DebugPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(320)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("QFrame { background: #1e1e1e; } QLabel { color: #ddd; font-family: Menlo; font-size: 12px; }")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        self._title = self._add_label(layout, "DOG DETECTOR DEBUG", bold=True)
        self._sep1 = self._add_sep(layout)
        self._fps = self._add_label(layout, "FPS: --")
        self._inference = self._add_label(layout, "Inference: --")
        self._detections = self._add_label(layout, "Detections: 0")
        self._sep2 = self._add_sep(layout)
        self._roi_info = self._add_label(layout, "ROI: not set")
        self._tracker = self._add_label(layout, "Tracker: --")
        self._tracks = self._add_label(layout, "Tracks: --")
        self._sep3 = self._add_sep(layout)
        self._script = self._add_label(layout, "Scripts: idle")
        self._counters = self._add_label(layout, "Frames: 0 | Inferences: 0")
        self._web = self._add_label(layout, "Web: stopped")
        self._sep4 = self._add_sep(layout)

        self._event_label = self._add_label(layout, "EVENT LOG", bold=True)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(300)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        self._log_widget = QLabel("")
        self._log_widget.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._log_widget.setWordWrap(True)
        self._log_widget.setStyleSheet("color: #aaa; font-family: Menlo; font-size: 11px;")
        scroll.setWidget(self._log_widget)
        layout.addWidget(scroll)

        layout.addStretch()

    def _add_label(self, layout, text, bold=False):
        lbl = QLabel(text)
        if bold:
            lbl.setStyleSheet("color: #fff; font-weight: bold; font-size: 13px;")
        layout.addWidget(lbl)
        return lbl

    def _add_sep(self, layout):
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #444;")
        layout.addWidget(sep)
        return sep

    def update_state(self, state: AppState):
        t = state.timings
        self._fps.setText(f"Camera FPS: {t.get('camera_fps', 0):.1f} | Inference FPS: {t.get('inference_fps', 0):.1f}")
        self._inference.setText(f"Inference: {t.get('inference_ms', 0):.1f}ms | Render: {t.get('render_ms', 0):.1f}ms")

        dets = state.latest_detections
        if dets:
            confs = ", ".join(f"{d.confidence:.0%}" for d in dets)
            self._detections.setText(f"Dogs: {len(dets)} [{confs}]")
        else:
            self._detections.setText("Dogs: 0")

        pts = state.roi_points or []
        if pts:
            self._roi_info.setText(f"ROI: {len(pts)} pts | valid")
        else:
            self._roi_info.setText("ROI: not set")

        ts = state.tracker_state
        if ts:
            inside = "IN" if ts.get("dog_inside") else "OUT"
            since = time.time() - ts.get("last_change_time", time.time())
            self._tracker.setText(
                f"Dog: {inside} ({since:.0f}s ago) | E:{ts.get('enter_count', 0)} L:{ts.get('leave_count', 0)}"
            )
            tracks = ts.get("tracks", {})
            if tracks:
                lines = []
                for tid, t in tracks.items():
                    status = "IN" if t["confirmed"] else "out"
                    lines.append(f"  #{tid}: {status} roi+{t['in_roi_count']} abs+{t['absent_count']}")
                self._tracks.setText("Tracks:\n" + "\n".join(lines))
            else:
                self._tracks.setText("Tracks: none")

        self._counters.setText(f"Frames: {state.frame_count} | Inferences: {state.inference_count}")
        self._web.setText(f"Web: {state.web_clients} clients")

        log_lines = list(state.event_log)[:50]
        self._log_widget.setText("\n".join(log_lines))
