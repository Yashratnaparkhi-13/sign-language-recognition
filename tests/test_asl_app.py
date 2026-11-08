from __future__ import annotations

import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PySide6.QtCore import QObject, Signal
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication

from asl_app.camera_worker import CameraWorker
from asl_app.main_window import MainWindow
from asl_app.model import EXPECTED_CLASSES, LandmarkClassifier
from asl_app.sentence_manager import SentenceManager
from asl_app.speech_worker import SpeechWorker
from asl_app.stabilizer import PredictionStabilizer

PROJECT_DIR = Path(__file__).resolve().parents[1]


class FakeSpeechController(QObject):
    status_changed = Signal(bool, str)
    error_occurred = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.spoken: list[str] = []
        self.muted = False

    def start(self) -> None:
        pass

    def isRunning(self) -> bool:
        return False

    def speak(self, text: str) -> None:
        if not self.muted:
            self.spoken.append(text)

    def set_muted(self, muted: bool) -> None:
        self.muted = muted

    def shutdown(self) -> None:
        pass


class FakeCapture:
    def __init__(self, _camera_index: int, read_success: bool = False) -> None:
        self.read_success = read_success
        self.released = False

    def isOpened(self) -> bool:
        return True

    def read(self):
        if self.read_success:
            return True, np.zeros((32, 32, 3), dtype=np.uint8)
        return False, None

    def release(self) -> None:
        self.released = True


class FakeDetector:
    def __init__(self, _task_path: Path) -> None:
        self.closed = False

    def detect(self, _frame):
        return []

    def close(self) -> None:
        self.closed = True


class FakeClassifier:
    def __init__(self, _checkpoint_path: Path) -> None:
        pass


class QtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])


class ModelTests(unittest.TestCase):
    def test_active_checkpoint_has_exact_28_class_mapping(self) -> None:
        classifier = LandmarkClassifier(PROJECT_DIR / "sign_landmark.pt")
        self.assertEqual(len(classifier.classes), 28)
        self.assertEqual(classifier.classes, EXPECTED_CLASSES)

    def test_inference_returns_a_mapped_class_and_probability(self) -> None:
        classifier = LandmarkClassifier(PROJECT_DIR / "sign_landmark.pt")
        landmarks = [
            SimpleNamespace(x=index / 21, y=index / 42, z=0.0)
            for index in range(21)
        ]
        label, confidence = classifier.predict(landmarks)
        self.assertIn(label, EXPECTED_CLASSES)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)


class StabilizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.stabilizer = PredictionStabilizer()

    def stabilize(self, label: str) -> str | None:
        accepted = None
        for _ in range(7):
            _, _, accepted = self.stabilizer.update(label, 0.95, True)
        return accepted

    def test_accepts_letter_only_once_while_gesture_is_held(self) -> None:
        self.assertEqual(self.stabilize("A"), "A")
        for _ in range(15):
            _, _, accepted = self.stabilizer.update("A", 0.95, True)
            self.assertIsNone(accepted)

    def test_accepts_transition_and_repeated_letter_after_hand_release(self) -> None:
        self.assertEqual(self.stabilize("Y"), "Y")
        self.assertEqual(self.stabilize("A"), "A")
        for _ in range(8):
            stable, _, _ = self.stabilizer.update(None, 0.0, False)
        self.assertIsNone(stable)
        self.assertEqual(self.stabilize("A"), "A")

    def test_space_and_delete_are_valid_model_gestures(self) -> None:
        self.assertEqual(self.stabilize("space"), "space")
        stabilizer = PredictionStabilizer()
        accepted = None
        for _ in range(7):
            _, _, accepted = stabilizer.update("del", 0.95, True)
        self.assertEqual(accepted, "del")


class SentenceTests(unittest.TestCase):
    def test_constructs_words_with_space_delete_and_replayable_text(self) -> None:
        manager = SentenceManager()
        for letter in "YASH":
            manager.apply(letter)
        completed = manager.apply("space")
        self.assertEqual(completed.text, "YASH ")
        self.assertEqual(completed.spoken_word, "YASH")
        manager.apply("Y")
        manager.apply("A")
        self.assertEqual(manager.text, "YASH YA")
        self.assertEqual(manager.apply("del").text, "YASH Y")

    def test_space_does_not_add_leading_or_duplicate_spaces(self) -> None:
        manager = SentenceManager()
        self.assertEqual(manager.apply("space").text, "")
        manager.apply("A")
        manager.apply("space")
        self.assertEqual(manager.apply("space").text, "A ")


class SpeechTests(unittest.TestCase):
    def test_async_speech_mute_and_replay(self) -> None:
        spoken: list[str] = []
        changed = threading.Condition()

        class FakeEngine:
            def setProperty(self, _name, _value) -> None:
                pass

            def say(self, text: str) -> None:
                with changed:
                    spoken.append(text)
                    changed.notify_all()

            def runAndWait(self) -> None:
                pass

            def stop(self) -> None:
                pass

        worker = SpeechWorker(engine_factory=FakeEngine)
        worker.start()
        worker.set_muted(True)
        worker.speak("SHOULD NOT PLAY")
        worker.set_muted(False)
        worker.speak("YASH")
        worker.speak("YASH")
        deadline = time.monotonic() + 3
        with changed:
            while len(spoken) < 2 and time.monotonic() < deadline:
                changed.wait(deadline - time.monotonic())
        worker.shutdown()
        worker.wait(3000)
        self.assertFalse(worker.isRunning())
        self.assertEqual(spoken, ["YASH", "YASH"])


class CameraWorkerTests(QtTests):
    def test_camera_disconnect_is_reported_and_resources_are_closed(self) -> None:
        capture = FakeCapture(0)
        detector = FakeDetector(Path("hand_landmarker.task"))
        worker = CameraWorker(
            Path("sign_landmark.pt"),
            Path("hand_landmarker.task"),
            capture_factory=lambda index: capture,
            classifier_factory=FakeClassifier,
            detector_factory=lambda path: detector,
        )
        status_spy = QSignalSpy(worker.camera_status_changed)
        error_spy = QSignalSpy(worker.error_occurred)
        worker.start()
        self.assertTrue(worker.wait(3000))
        statuses = [status_spy.at(index)[1] for index in range(status_spy.count())]
        self.assertIn("Camera disconnected", statuses)
        self.assertEqual(error_spy.count(), 1)
        self.assertTrue(capture.released)
        self.assertTrue(detector.closed)

    def test_requested_shutdown_releases_camera(self) -> None:
        capture = FakeCapture(0, read_success=True)
        detector = FakeDetector(Path("hand_landmarker.task"))
        worker = CameraWorker(
            Path("sign_landmark.pt"),
            Path("hand_landmarker.task"),
            capture_factory=lambda index: capture,
            classifier_factory=FakeClassifier,
            detector_factory=lambda path: detector,
        )
        worker.start()
        worker.request_stop()
        self.assertTrue(worker.wait(3000))
        self.assertTrue(capture.released)
        self.assertTrue(detector.closed)


class MainWindowTests(QtTests):
    def test_sentence_controls_voice_and_history(self) -> None:
        speech = FakeSpeechController()
        window = MainWindow(project_dir=PROJECT_DIR, speech_worker=speech)
        window._on_camera_status(False, "Camera unavailable")
        self.assertEqual(window.camera_view.text(), "Camera unavailable")
        for letter in "YASH":
            window._accept_gesture(letter)
        window._accept_gesture("space")
        self.assertEqual(window.sentence.text, "YASH ")
        self.assertEqual(speech.spoken, ["YASH"])
        self.assertEqual(window.history_list.count(), 5)

        window.speak_sentence()
        self.assertEqual(speech.spoken[-1], "YASH")
        window.toggle_voice()
        self.assertTrue(speech.muted)
        previous_count = len(speech.spoken)
        window.speak_sentence()
        self.assertEqual(len(speech.spoken), previous_count)
        window.toggle_voice()
        window.speak_sentence()
        self.assertEqual(speech.spoken[-1], "YASH")

        window.backspace()
        self.assertEqual(window.sentence.text, "YASH")
        window.clear_sentence()
        self.assertEqual(window.sentence.text, "")
        self.assertEqual(window.sentence_display.text(), "Your text will appear here")
        window.close()

    def test_closing_window_stops_active_camera_worker(self) -> None:
        speech = FakeSpeechController()
        capture = FakeCapture(0, read_success=True)
        detector = FakeDetector(Path("hand_landmarker.task"))

        def worker_factory(checkpoint_path, task_path, parent=None):
            return CameraWorker(
                checkpoint_path,
                task_path,
                capture_factory=lambda index: capture,
                classifier_factory=FakeClassifier,
                detector_factory=lambda path: detector,
                parent=parent,
            )

        window = MainWindow(
            project_dir=PROJECT_DIR,
            camera_worker_factory=worker_factory,
            speech_worker=speech,
        )
        window.start_camera()
        worker = window.camera_worker
        window.close()
        self.assertIsNotNone(worker)
        self.assertFalse(worker.isRunning())
        self.assertTrue(capture.released)
        self.assertTrue(detector.closed)


if __name__ == "__main__":
    unittest.main()
