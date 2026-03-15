from __future__ import annotations

from typing import Iterable, List

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
