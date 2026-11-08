from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable, Optional

import cv2
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from .hand_detector import HandDetector
from .model import LandmarkClassifier
from .stabilizer import PredictionStabilizer

LOGGER = logging.getLogger(__name__)


class CameraWorker(QThread):
    frame_ready = Signal(QImage)
    prediction_changed = Signal(object, float)
    gesture_accepted = Signal(str)
    camera_status_changed = Signal(bool, str)
    model_status_changed = Signal(bool, str)
    error_occurred = Signal(str)

    def __init__(
        self,
        checkpoint_path: Path,
        task_path: Path,
        camera_index: int = 0,
        capture_factory: Callable[[int], object] = cv2.VideoCapture,
        classifier_factory: Callable[[Path], object] = LandmarkClassifier,
        detector_factory: Callable[[Path], object] = HandDetector,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.checkpoint_path = checkpoint_path
        self.task_path = task_path
        self.camera_index = camera_index
        self._capture_factory = capture_factory
        self._classifier_factory = classifier_factory
        self._detector_factory = detector_factory
        self._stop_event = threading.Event()

    def request_stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        capture = None
        detector = None
        model_ready = False
        try:
            self.model_status_changed.emit(False, "Loading model")
            classifier = self._classifier_factory(self.checkpoint_path)
            model_ready = True
            self.model_status_changed.emit(True, "Model ready · 28 classes")
            detector = self._detector_factory(self.task_path)

            capture = self._capture_factory(self.camera_index)
            if not capture.isOpened():
                message = "Could not open camera. Check permissions and connection."
                self.camera_status_changed.emit(False, "Camera unavailable")
                self.error_occurred.emit(message)
                return

            self.camera_status_changed.emit(True, "Camera active")
            stabilizer = PredictionStabilizer()
            last_prediction: tuple[object, float] = (None, 0.0)

            while not self._stop_event.is_set():
                success, frame = capture.read()
                if not success:
                    if not self._stop_event.is_set():
                        self.camera_status_changed.emit(
                            False, "Camera disconnected"
                        )
                        self.error_occurred.emit(
                            "The camera stopped returning frames."
                        )
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                landmarks = detector.detect(rgb_frame)

                label: Optional[str] = None
                confidence = 0.0
                if landmarks:
                    label, confidence = classifier.predict(landmarks)
                    for landmark in landmarks:
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        cv2.circle(frame, (x, y), 4, (45, 205, 135), -1)

                stable, stable_confidence, accepted = stabilizer.update(
                    label,
                    confidence,
                    bool(landmarks),
                )
                current_prediction = (stable, stable_confidence)
                if current_prediction != last_prediction:
                    self.prediction_changed.emit(stable, stable_confidence)
                    last_prediction = current_prediction
                if accepted is not None:
                    self.gesture_accepted.emit(accepted)

                display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, _ = display.shape
                image = QImage(
                    display.data,
                    width,
                    height,
                    display.strides[0],
                    QImage.Format.Format_RGB888,
                ).copy()
                self.frame_ready.emit(image)

        except Exception as error:
            LOGGER.exception("Camera worker failed")
            if not model_ready:
                self.model_status_changed.emit(False, "Model unavailable")
            self.error_occurred.emit(str(error))
            self.camera_status_changed.emit(False, "Camera error")
        finally:
            if capture is not None:
                capture.release()
            if detector is not None:
                detector.close()
            if self._stop_event.is_set():
                self.camera_status_changed.emit(False, "Camera stopped")
