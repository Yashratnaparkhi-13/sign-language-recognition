from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
from asl_app.model import EXPECTED_CLASSES, LandmarkClassifier
from asl_app.sentence_manager import SentenceManager
from asl_app.stabilizer import PredictionStabilizer
from web_app.recognition.processor import ASLVideoProcessor
from web_app.recognition.state import RecognitionState

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeDetector:
    def __init__(self, _task_path: Path) -> None:
        self.closed = False

    def detect(self, _rgb_frame):
        return [
            SimpleNamespace(x=index / 21, y=index / 42, z=0.0)
            for index in range(21)
        ]

    def close(self) -> None:
        self.closed = True


class FakeClassifier:
    def predict(self, _landmarks):
        return "A", 0.96


class WebStateTests(unittest.TestCase):
    def test_existing_model_mapping_and_asset_paths(self) -> None:
        self.assertEqual(len(EXPECTED_CLASSES), 28)
        self.assertEqual(EXPECTED_CLASSES[:26], tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        self.assertEqual(EXPECTED_CLASSES[-2:], ("del", "space"))
        self.assertTrue((PROJECT_ROOT / "sign_landmark.pt").is_file())
        self.assertTrue((PROJECT_ROOT / "hand_landmarker.task").is_file())

    def test_existing_checkpoint_loads_in_web_classifier(self) -> None:
        classifier = LandmarkClassifier(PROJECT_ROOT / "sign_landmark.pt")
        self.assertEqual(len(classifier.classes), 28)
        self.assertEqual(classifier.classes, EXPECTED_CLASSES)

    def test_classifier_rejects_missing_checkpoint(self) -> None:
        with self.assertRaises(FileNotFoundError):
            LandmarkClassifier(PROJECT_ROOT / "missing-sign-model.pt")

    def test_sentence_spell_space_delete_and_replay_word_source(self) -> None:
        state = RecognitionState()
        for letter in "YASH":
            state.accept(letter)
        state.accept("space")
        snapshot = state.snapshot()
        self.assertEqual(snapshot.sentence, "YASH ")
        self.assertEqual(snapshot.last_completed_word, "YASH")
        self.assertEqual(snapshot.word_event_id, 1)
        state.accept("Y")
        state.accept("A")
        state.accept("S")
        state.accept("H")
        state.accept("space")
        self.assertEqual(state.snapshot().word_event_id, 2)

        state.accept("del")
        self.assertEqual(state.snapshot().sentence, "YASH YASH")
        self.assertEqual(
            state.snapshot().recent,
            ("Y", "A", "S", "H", "SPACE", "Y", "A", "S", "H", "SPACE", "DELETE"),
        )

    def test_sentence_manager_produces_hello_with_repeated_letter(self) -> None:
        manager = SentenceManager()
        for letter in "HELLO":
            manager.apply(letter)
        self.assertEqual(manager.text, "HELLO")

    def test_gesture_flow_builds_repeated_letter_word(self) -> None:
        state = RecognitionState()
        stabilizer = state.stabilizer

        for letter in "HEL":
            for _ in range(7):
                _, _, accepted = stabilizer.update(letter, 0.96, True)
                if accepted:
                    state.accept(accepted)
        for _ in range(8):
            stabilizer.update(None, 0.0, False)
        for letter in "LO":
            for _ in range(7):
                _, _, accepted = stabilizer.update(letter, 0.96, True)
                if accepted:
                    state.accept(accepted)
        self.assertEqual(state.snapshot().sentence, "HELLO")

    def test_stabilizer_blocks_held_letter_and_rearms_after_release(self) -> None:
        stabilizer = PredictionStabilizer()
        accepted = []
        for _ in range(20):
            _, _, event = stabilizer.update("A", 0.95, True)
            if event:
                accepted.append(event)
        self.assertEqual(accepted, ["A"])

        for _ in range(8):
            stabilizer.update(None, 0.0, False)
        accepted = []
        for _ in range(7):
            _, _, event = stabilizer.update("A", 0.95, True)
            if event:
                accepted.append(event)
        self.assertEqual(accepted, ["A"])

    def test_state_clear_and_backspace(self) -> None:
        state = RecognitionState()
        state.accept("A")
        state.accept("B")
        state.backspace()
        self.assertEqual(state.snapshot().sentence, "A")
        state.clear()
        self.assertEqual(state.snapshot().sentence, "")
        self.assertEqual(state.snapshot().recent, ())

    def test_continuous_frame_callback_accepts_held_gesture_once(self) -> None:
        state = RecognitionState()
        detector = FakeDetector(PROJECT_ROOT / "hand_landmarker.task")
        processor = ASLVideoProcessor(
            state,
            FakeClassifier(),
            PROJECT_ROOT / "hand_landmarker.task",
            detector_factory=lambda path: detector,
        )
        for _ in range(20):
            output = processor.recv(
                av.VideoFrame.from_ndarray(
                    np.zeros((64, 80, 3), dtype=np.uint8),
                    format="bgr24",
                )
            )
            self.assertEqual(output.width, 80)

        snapshot = state.snapshot()
        self.assertEqual(snapshot.sentence, "A")
        self.assertEqual(snapshot.processed_frames, 20)
        processor.on_ended()
        self.assertTrue(detector.closed)

    def test_frame_throttle_keeps_video_frames_but_limits_inference(self) -> None:
        state = RecognitionState()
        state.stabilizer.required_votes = 1
        detector = FakeDetector(PROJECT_ROOT / "hand_landmarker.task")
        processor = ASLVideoProcessor(
            state,
            FakeClassifier(),
            PROJECT_ROOT / "hand_landmarker.task",
            frame_interval=0.05,
            detector_factory=lambda path: detector,
        )
        frame = av.VideoFrame.from_ndarray(
            np.zeros((64, 80, 3), dtype=np.uint8),
            format="bgr24",
        )
        first = processor.recv(frame)
        second = processor.recv(frame)

        self.assertEqual(first.width, 80)
        self.assertEqual(second.width, 80)
        self.assertEqual(state.snapshot().processed_frames, 1)
        self.assertEqual(state.snapshot().sentence, "A")
        processor.on_ended()

    def test_synchronous_latest_frame_mode_is_used_for_stream(self) -> None:
        app_source = (PROJECT_ROOT / "web_app" / "app.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("async_processing=False", app_source)
        self.assertIn("video_receiver_size=1", app_source)
        self.assertIn('"frameRate": {"ideal": 24, "max": 30}', app_source)


if __name__ == "__main__":
    unittest.main()
