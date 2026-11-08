from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .camera_worker import CameraWorker
from .sentence_manager import SentenceManager
from .speech_worker import SpeechWorker

LOGGER = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(
        self,
        project_dir: Optional[Path] = None,
        camera_worker_factory=CameraWorker,
        speech_worker: Optional[SpeechWorker] = None,
    ) -> None:
        super().__init__()
        self.project_dir = project_dir or Path(__file__).resolve().parent.parent
        self.camera_worker_factory = camera_worker_factory
        self.camera_worker: Optional[CameraWorker] = None
        self.speech_worker = speech_worker or SpeechWorker(parent=self)
        self.sentence = SentenceManager()
        self.voice_enabled = True
        self._last_frame: Optional[QPixmap] = None
        self.setWindowTitle("Signa · ASL Recognition")
        self.setMinimumSize(1040, 700)
        self.resize(1240, 820)
        self._build_ui()
        self._apply_theme()
        self.speech_worker.status_changed.connect(self._on_voice_status)
        self.speech_worker.error_occurred.connect(self._show_error)
        self.speech_worker.start()

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(28, 22, 28, 24)
        root_layout.setSpacing(18)

        header = QHBoxLayout()
        title_block = QVBoxLayout()
        title = QLabel("Signa")
        title.setObjectName("appTitle")
        subtitle = QLabel("ASL alphabet recognition & speech")
        subtitle.setObjectName("mutedText")
        title_block.addWidget(title)
        title_block.addWidget(subtitle)
        header.addLayout(title_block)
        header.addStretch(1)

        status_block = QVBoxLayout()
        self.model_status = QLabel("Model · Not loaded")
        self.model_status.setObjectName("statusPill")
        self.camera_status = QLabel("Camera · Stopped")
        self.camera_status.setObjectName("statusPill")
        status_block.addWidget(self.model_status, alignment=Qt.AlignmentFlag.AlignRight)
        status_block.addWidget(self.camera_status, alignment=Qt.AlignmentFlag.AlignRight)
        header.addLayout(status_block)
        root_layout.addLayout(header)

        content = QHBoxLayout()
        content.setSpacing(18)
        content.addWidget(self._build_camera_panel(), 3)
        content.addWidget(self._build_recognition_panel(), 2)
        root_layout.addLayout(content, 1)

        controls = QHBoxLayout()
        controls.setSpacing(10)
        self.start_button = QPushButton("Start camera")
        self.start_button.setObjectName("primaryButton")
        self.start_button.clicked.connect(self.start_camera)
        self.stop_button = QPushButton("Stop camera")
        self.stop_button.clicked.connect(self.stop_camera)
        self.stop_button.setEnabled(False)
        self.voice_button = QPushButton("Voice: ON")
        self.voice_button.clicked.connect(self.toggle_voice)
        self.speak_button = QPushButton("Speak / replay sentence")
        self.speak_button.clicked.connect(self.speak_sentence)
        self.backspace_button = QPushButton("Backspace")
        self.backspace_button.clicked.connect(self.backspace)
        self.clear_button = QPushButton("Clear sentence")
        self.clear_button.setObjectName("dangerButton")
        self.clear_button.clicked.connect(self.clear_sentence)
        for button in (
            self.start_button,
            self.stop_button,
            self.voice_button,
            self.speak_button,
            self.backspace_button,
            self.clear_button,
        ):
            controls.addWidget(button)
        root_layout.addLayout(controls)

        self.status_message = QLabel(
            "Hold a sign steadily; lift your hand to re-arm the same letter."
        )
        self.status_message.setObjectName("mutedText")
        root_layout.addWidget(self.status_message)
        self.setCentralWidget(root)

    def _build_camera_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)
        heading = QLabel("LIVE CAMERA")
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)
        self.camera_view = QLabel("Camera is stopped")
        self.camera_view.setObjectName("cameraView")
        self.camera_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_view.setMinimumSize(480, 360)
        self.camera_view.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        layout.addWidget(self.camera_view, 1)
        note = QLabel("Mirrored preview · Hand landmarks highlighted")
        note.setObjectName("mutedText")
        layout.addWidget(note)
        return panel

    def _build_recognition_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(14)

        current_heading = QLabel("CURRENT SIGN")
        current_heading.setObjectName("sectionTitle")
        layout.addWidget(current_heading)
        self.current_sign = QLabel("—")
        self.current_sign.setObjectName("currentSign")
        self.current_sign.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.current_sign)
        self.confidence_label = QLabel("Show a hand sign to begin")
        self.confidence_label.setObjectName("mutedText")
        self.confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.confidence_label)

        sentence_heading = QLabel("ACCUMULATED TEXT")
        sentence_heading.setObjectName("sectionTitle")
        layout.addWidget(sentence_heading)
        self.sentence_display = QLabel("")
        self.sentence_display.setObjectName("sentenceDisplay")
        self.sentence_display.setWordWrap(True)
        self.sentence_display.setMinimumHeight(92)
        self.sentence_display.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        layout.addWidget(self.sentence_display)

        history_heading = QLabel("RECOGNITION HISTORY")
        history_heading.setObjectName("sectionTitle")
        layout.addWidget(history_heading)
        self.history_list = QListWidget()
        self.history_list.setObjectName("historyList")
        layout.addWidget(self.history_list, 1)
        return panel

    def _apply_theme(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #0b1220;
                color: #e6edf7;
                font-family: "Avenir Next", "Segoe UI", sans-serif;
                font-size: 14px;
            }
            QFrame#panel {
                background: #111c2e;
                border: 1px solid #22314a;
                border-radius: 16px;
            }
            QLabel#appTitle { font-size: 30px; font-weight: 700; color: #f4f8ff; }
            QLabel#mutedText { color: #93a4bd; }
            QLabel#sectionTitle {
                color: #89a9d8;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 1px;
            }
            QLabel#statusPill {
                color: #b7c8df;
                background: #14233a;
                border: 1px solid #263d5e;
                border-radius: 9px;
                padding: 6px 10px;
            }
            QLabel#cameraView {
                background: #080e18;
                border: 1px solid #26354d;
                border-radius: 12px;
                color: #71819a;
                font-size: 17px;
            }
            QLabel#currentSign {
                background: #152a48;
                border: 1px solid #2c5e9d;
                border-radius: 14px;
                color: #67b4ff;
                font-size: 88px;
                font-weight: 700;
                min-height: 150px;
            }
            QLabel#sentenceDisplay {
                background: #0b1525;
                border: 1px solid #243b5a;
                border-radius: 12px;
                color: #f4f8ff;
                font-size: 24px;
                font-weight: 600;
                padding: 14px;
            }
            QListWidget#historyList {
                background: #0b1525;
                border: 1px solid #24364f;
                border-radius: 10px;
                padding: 5px;
            }
            QPushButton {
                background: #1a2a42;
                border: 1px solid #2b4262;
                border-radius: 9px;
                color: #dce8f8;
                font-weight: 600;
                min-height: 40px;
                padding: 0 14px;
            }
            QPushButton:hover { background: #243b5b; }
            QPushButton:disabled { color: #64748b; background: #131d2b; }
            QPushButton#primaryButton {
                background: #1768b5;
                border-color: #2787df;
                color: white;
            }
            QPushButton#primaryButton:hover { background: #217ed0; }
            QPushButton#dangerButton { color: #ffb5b5; border-color: #6a3945; }
            """
        )

    def start_camera(self) -> None:
        if self.camera_worker is not None and self.camera_worker.isRunning():
            return
        self.current_sign.setText("—")
        self.confidence_label.setText("Waiting for a stable prediction")
        self.camera_view.setText("Opening camera…")
        self.camera_status.setText("Camera · Starting")
        self.model_status.setText("Model · Loading")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        worker = self.camera_worker_factory(
            self.project_dir / "sign_landmark.pt",
            self.project_dir / "hand_landmarker.task",
            parent=self,
        )
        worker.frame_ready.connect(self._show_frame)
        worker.prediction_changed.connect(self._show_prediction)
        worker.gesture_accepted.connect(self._accept_gesture)
        worker.camera_status_changed.connect(self._on_camera_status)
        worker.model_status_changed.connect(self._on_model_status)
        worker.error_occurred.connect(self._show_error)
        worker.finished.connect(self._on_camera_finished)
        self.camera_worker = worker
        worker.start()

    def stop_camera(self) -> None:
        if self.camera_worker is not None and self.camera_worker.isRunning():
            self.camera_status.setText("Camera · Stopping")
            self.stop_button.setEnabled(False)
            self.camera_worker.request_stop()

    def toggle_voice(self) -> None:
        self.voice_enabled = not self.voice_enabled
        self.voice_button.setText(
            f"Voice: {'ON' if self.voice_enabled else 'OFF'}"
        )
        self.speech_worker.set_muted(not self.voice_enabled)

    def speak_sentence(self) -> None:
        text = self.sentence.text.strip()
        if not text:
            self.status_message.setText("There is no sentence to speak yet.")
            return
        if not self.voice_enabled:
            self.status_message.setText("Voice is muted.")
            return
        self.speech_worker.speak(text)

    def clear_sentence(self) -> None:
        self.sentence.clear()
        self._refresh_sentence()

    def backspace(self) -> None:
        if self.sentence.text:
            self.sentence.text = self.sentence.text[:-1]
            self._refresh_sentence()

    def _show_frame(self, image) -> None:
        pixmap = QPixmap.fromImage(image)
        self._last_frame = pixmap
        self._update_camera_pixmap()

    def _update_camera_pixmap(self) -> None:
        if self._last_frame is None:
            return
        self.camera_view.setPixmap(
            self._last_frame.scaled(
                self.camera_view.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_camera_pixmap()

    def _show_prediction(self, label, confidence: float) -> None:
        if label is None:
            self.current_sign.setText("—")
            self.confidence_label.setText("No hand detected · Ready")
            return
        display = (
            "SPACE" if label == "space"
            else "DELETE" if label == "del"
            else label
        )
        self.current_sign.setText(display)
        self.confidence_label.setText(f"Stable prediction · {confidence:.1%}")

    def _accept_gesture(self, label: str) -> None:
        update = self.sentence.apply(label)
        self._refresh_sentence()
        self.history_list.insertItem(0, update.action)
        while self.history_list.count() > 12:
            self.history_list.takeItem(self.history_list.count() - 1)
        if update.spoken_word and self.voice_enabled:
            self.speech_worker.speak(update.spoken_word)
            self.status_message.setText(f"Speaking completed word: {update.spoken_word}")
        else:
            self.status_message.setText(f"Accepted {update.action}")

    def _refresh_sentence(self) -> None:
        self.sentence_display.setText(self.sentence.text or "Your text will appear here")

    def _on_camera_status(self, active: bool, message: str) -> None:
        self.camera_status.setText(f"Camera · {message.removeprefix('Camera ')}")
        if active:
            self.camera_view.setText("")
        else:
            self._last_frame = None
            self.camera_view.setPixmap(QPixmap())
            self.camera_view.setText(message)

    def _on_model_status(self, ready: bool, message: str) -> None:
        self.model_status.setText(f"Model · {message}")

    def _on_voice_status(self, ready: bool, message: str) -> None:
        if not ready:
            self.voice_enabled = False
            self.voice_button.setText("Voice: OFF")
            self.voice_button.setEnabled(False)
        self.status_message.setText(message)

    def _show_error(self, message: str) -> None:
        LOGGER.error("%s", message)
        self.status_message.setText(message)

    def _on_camera_finished(self) -> None:
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if self.camera_status.text() in (
            "Camera · Starting",
            "Camera · active",
        ):
            self.camera_status.setText("Camera · Stopped")

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.camera_worker is not None and self.camera_worker.isRunning():
            self.camera_worker.request_stop()
            self.camera_worker.wait()
        if self.speech_worker.isRunning():
            self.speech_worker.shutdown()
            self.speech_worker.wait()
        event.accept()
