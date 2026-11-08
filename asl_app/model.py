from __future__ import annotations

from pathlib import Path
from string import ascii_uppercase
from typing import Sequence

import numpy as np
import torch


EXPECTED_CLASSES = tuple(ascii_uppercase) + ("del", "space")


class LandmarkMLP(torch.nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(63, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, n_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def normalize_landmarks(landmarks: Sequence[object]) -> np.ndarray:
    points = np.array(
        [[point.x, point.y, point.z] for point in landmarks],
        dtype=np.float32,
    )
    if points.shape != (21, 3):
        raise ValueError(f"Expected 21 hand landmarks, got {points.shape}.")
    points = points - points[0]
    scale = np.linalg.norm(points, axis=1).max()
    return (points / (scale + 1e-6)).astype(np.float32)


class LandmarkClassifier:
    def __init__(self, checkpoint_path: Path) -> None:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        classes = tuple(str(name) for name in checkpoint.get("classes", ()))
        if classes != EXPECTED_CLASSES:
            raise ValueError(
                "Checkpoint classes do not match the expected A-Z, del, "
                f"space mapping. Found: {classes}"
            )

        state = checkpoint.get("model")
        if not isinstance(state, dict):
            raise ValueError("Checkpoint does not contain a model state dict.")
        output_weights = state.get("net.9.weight")
        if output_weights is None or output_weights.shape[0] != len(classes):
            raise ValueError(
                "Checkpoint output layer does not match its class mapping."
            )

        self.classes = classes
        self.model = LandmarkMLP(len(classes))
        self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, landmarks: Sequence[object]) -> tuple[str, float]:
        normalized = normalize_landmarks(landmarks)
        input_tensor = torch.from_numpy(normalized.reshape(1, 63))
        with torch.no_grad():
            probabilities = torch.softmax(self.model(input_tensor), dim=1)
            confidence, predicted_index = torch.max(probabilities, dim=1)
        return (
            self.classes[int(predicted_index.item())],
            float(confidence.item()),
        )
