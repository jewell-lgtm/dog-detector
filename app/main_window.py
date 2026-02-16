import logging
import time

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QImage, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QStatusBar,
    QToolBar,
    QWidget,
)

from app.camera import CameraThread
from app.config import Config
from app.debug_panel import DebugPanel
from app.detector import DogDetector
from app.overlay import OverlayPainter
from app.roi import ROI
from app.script_runner import ScriptRunner
from app.state import AppState
from app.tracker import Tracker

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self, config: Config, state: AppState):
        super().__init__()
        self.config = config
        self.state = state
        self.setWindowTitle("Dog Detector")
        self.setMinimumSize(1000, 600)

        # components
        self.roi = ROI()
        if config.roi_points:
            self.roi.set_points([tuple(p) for p in config.roi_points])
            self.state.roi_points = self.roi.points
        self.detector = DogDetector(confidence=config.confidence)
        self.tracker = Tracker(enter_frames=config.enter_frames, leave_frames=config.leave_frames)
        self.overlay = OverlayPainter()
        self.script_runner = ScriptRunner(cooldown=config.cooldown)
        self._drawing_roi = False
        self._last_detections: list = []

        # frame timing
        self._frame_count = 0
        self._inference_count = 0
        self._fps_time = time.time()
        self._fps_frames = 0
        self._inf_time = time.time()
        self._inf_frames = 0

        # layout
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        self._video_label = QLabel()
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setStyleSheet("background: black;")
        self._video_label.setMinimumSize(640, 480)
        self._video_label.mousePressEvent = self._on_video_click
        layout.addWidget(self._video_label, stretch=1)

        self._debug_panel = DebugPanel()
        layout.addWidget(self._debug_panel)

        # toolbar
        toolbar = QToolBar("Controls")
        self.addToolBar(toolbar)

        draw_roi_action = QAction("Draw ROI", self)
        draw_roi_action.setCheckable(True)
        draw_roi_action.toggled.connect(self._toggle_roi_drawing)
        toolbar.addAction(draw_roi_action)
        self._draw_roi_action = draw_roi_action

        clear_roi_action = QAction("Clear ROI", self)
        clear_roi_action.triggered.connect(self._clear_roi)
        toolbar.addAction(clear_roi_action)

        toolbar.addSeparator()

        enter_script_action = QAction("Enter Script...", self)
        enter_script_action.triggered.connect(lambda: self._pick_script("enter"))
        toolbar.addAction(enter_script_action)

        leave_script_action = QAction("Leave Script...", self)
        leave_script_action.triggered.connect(lambda: self._pick_script("leave"))
        toolbar.addAction(leave_script_action)

        self.setStatusBar(QStatusBar())

        # camera
        self._camera = CameraThread(device=config.camera_device)
        self._camera.frame_ready.connect(self._on_frame)
        self._camera.start()

    def _on_frame(self, frame: np.ndarray):
        try:
            self._process_frame(frame)
        except Exception:
            logger.exception("_on_frame crashed")

    def _process_frame(self, frame: np.ndarray):
        self._frame_count += 1
        self.state.frame_count = self._frame_count

        # fps calc
        self._fps_frames += 1
        now = time.time()
        elapsed = now - self._fps_time
        if elapsed >= 1.0:
            self.state.timings["camera_fps"] = self._fps_frames / elapsed
            self._fps_frames = 0
            self._fps_time = now

        # check for web ROI updates
        web_roi = self.state.consume_roi_update()
        if web_roi is not None:
            self.roi.set_points(web_roi)
            self._save_roi()

        # check for web triggers
        trig_enter, trig_leave = self.state.consume_triggers()
        if trig_enter and self.config.enter_script:
            fired = self.script_runner.run(self.config.enter_script)
            if fired:
                self.state.log_event("MANUAL ENTER TRIGGER")
        if trig_leave and self.config.leave_script:
            fired = self.script_runner.run(self.config.leave_script)
            if fired:
                self.state.log_event("MANUAL LEAVE TRIGGER")

        # detect every Nth frame
        detections = self._last_detections
        inference_ms = self.state.timings.get("inference_ms", 0)
        if self._frame_count % self.config.inference_interval == 0:
            t0 = time.time()
            detections = self.detector.detect(frame)
            inference_ms = (time.time() - t0) * 1000
            self._last_detections = detections
            self._inference_count += 1
            self.state.inference_count = self._inference_count

            self._inf_frames += 1
            inf_elapsed = now - self._inf_time
            if inf_elapsed >= 1.0:
                self.state.timings["inference_fps"] = self._inf_frames / inf_elapsed
                self._inf_frames = 0
                self._inf_time = now

            # track
            entered, left = self.tracker.update(detections, self.roi)
            if entered:
                self.state.log_event("DOG ENTERED")
                if self.config.enter_script:
                    fired = self.script_runner.run(self.config.enter_script)
                    if fired:
                        self.state.log_event(f"Fired enter script")
            if left:
                self.state.log_event("DOG LEFT")
                if self.config.leave_script:
                    fired = self.script_runner.run(self.config.leave_script)
                    if fired:
                        self.state.log_event(f"Fired leave script")

        # draw
        t0 = time.time()
        annotated = self.overlay.draw(frame, detections, self.roi, self.tracker.state.dog_inside)

        # draw ROI points while drawing
        if self._drawing_roi and self.roi.points:
            for pt in self.roi.points:
                cv2.circle(annotated, pt, 5, (255, 255, 0), -1)

        render_ms = (time.time() - t0) * 1000

        # update shared state
        self.state.update_frame(annotated, detections, self.tracker.as_dict(), inference_ms, render_ms)

        # display
        h, w, ch = annotated.shape
        img = QImage(annotated.data, w, h, ch * w, QImage.Format.Format_BGR888)
        scaled = QPixmap.fromImage(img).scaled(
            self._video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self._video_label.setPixmap(scaled)

        self._debug_panel.update_state(self.state)

    def _on_video_click(self, event):
        if not self._drawing_roi:
            return
        # map click to frame coords
        pixmap = self._video_label.pixmap()
        if pixmap is None:
            return
        label_size = self._video_label.size()
        pm_size = pixmap.size()
        offset_x = (label_size.width() - pm_size.width()) // 2
        offset_y = (label_size.height() - pm_size.height()) // 2
        x = event.pos().x() - offset_x
        y = event.pos().y() - offset_y
        if x < 0 or y < 0 or x >= pm_size.width() or y >= pm_size.height():
            return

        # scale to frame coords
        frame = self.state.latest_annotated_frame
        if frame is None:
            return
        fh, fw = frame.shape[:2]
        fx = int(x * fw / pm_size.width())
        fy = int(y * fh / pm_size.height())

        if event.button() == Qt.MouseButton.RightButton:
            # close polygon
            self._drawing_roi = False
            self._draw_roi_action.setChecked(False)
            self._save_roi()
            self.state.log_event(f"ROI set ({len(self.roi.points)} pts)")
        else:
            self.roi.add_point(fx, fy)

    def _toggle_roi_drawing(self, checked):
        self._drawing_roi = checked
        if checked:
            self.roi.clear()
            self.statusBar().showMessage("Click to add ROI points. Right-click to finish.")
        else:
            self.statusBar().clearMessage()

    def _clear_roi(self):
        self.roi.clear()
        self.state.roi_points = []
        self.config.roi_points = []
        self.config.save()
        self.state.log_event("ROI cleared")

    def _save_roi(self):
        self.state.roi_points = self.roi.points
        self.config.roi_points = [list(p) for p in self.roi.points]
        self.config.save()

    def _pick_script(self, which: str):
        path, _ = QFileDialog.getOpenFileName(self, f"Select {which} AppleScript", "", "AppleScript (*.scpt *.applescript)")
        if not path:
            return
        if which == "enter":
            self.config.enter_script = path
        else:
            self.config.leave_script = path
        self.config.save()
        self.state.log_event(f"Set {which} script: {path}")

    def closeEvent(self, event):
        self._camera.stop()
        self.config.save()
        super().closeEvent(event)
