from __future__ import annotations

from typing import Iterable, List
import json
import os
import tempfile

from PIL import Image
import torch
from torchvision import transforms

from MIREIA.perception.e2e_model import E2ERiskPredictor


class E2EInference:
    def __init__(
        self,
        model: E2ERiskPredictor,
        device: torch.device,
        image_size: tuple[int, int] = (512, 512),
    ):
        self.model = model.to(device)
        self.device = device
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ]
        )

    def load_weights(self, checkpoint_path: str) -> None:
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    def predict_from_paths(self, image_paths: Iterable[str]) -> float:
        seq_tensor = torch.stack([self._load_image(path) for path in image_paths], dim=0)
        return self.predict_from_tensor(seq_tensor)

    def predict_from_tensor(self, seq_tensor: torch.Tensor) -> float:
        self.model.eval()
        if seq_tensor.ndim == 4:
            seq_tensor = seq_tensor.unsqueeze(0)
        seq_tensor = seq_tensor.to(self.device)
        with torch.no_grad():
            pred = self.model(seq_tensor)
        return float(pred.squeeze().item())

    def _load_image(self, path: str) -> torch.Tensor:
        with Image.open(path) as img:
            img = img.convert("RGB")
            return self.transform(img)


def append_model_risk_to_jsonl(
    jsonl_path: str,
    checkpoint_path: str,
    seq_len: int,
    device: torch.device | None = None,
    image_root: str | None = None,
) -> str:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_root = image_root or os.path.dirname(jsonl_path)
    model = E2ERiskPredictor()
    infer = E2EInference(model=model, device=device)
    infer.load_weights(checkpoint_path)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    def _resolve(path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(image_root, path))

    for i in range(len(records)):
        if i < seq_len - 1:
            records[i]["model_risk"] = None
            continue
        seq_paths = [
            _resolve(records[j].get("rgb_image_path", ""))
            for j in range(i - seq_len + 1, i + 1)
        ]
        if any(not p or not os.path.exists(p) for p in seq_paths):
            records[i]["model_risk"] = None
            continue
        records[i]["model_risk"] = infer.predict_from_paths(seq_paths)

    fd, tmp_path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    with open(tmp_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    os.replace(tmp_path, jsonl_path)
    return jsonl_path
