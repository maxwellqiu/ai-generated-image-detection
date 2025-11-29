"""
Dataset utilities for real vs AI-generated image classification.

Assumes directory structure:

    root/
      real/
      fake/

Label convention:
    0 = real
    1 = fake
"""

import os
from typing import Optional, Callable, List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_default_transform(img_size: int = 224) -> Callable:
    """Standard ImageNet-style transform for ResNet."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class RealFakeDataset(Dataset):
    """
    Simple image folder dataset for:

        root/
          real/
          fake/

    where:
      - "real" is mapped to label 0
      - "fake" is mapped to label 1
    """

    def __init__(self, root: str, transform: Optional[Callable] = None):
        self.root = root
        self.classes = ["real", "fake"]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples: List[Tuple[str, int]] = []

        # Scan directory and collect (path, label)
        for cls in self.classes:
            folder = os.path.join(root, cls)
            if not os.path.isdir(folder):
                continue
            label = self.class_to_idx[cls]
            for fname in os.listdir(folder):
                path = os.path.join(folder, fname)
                if os.path.isfile(path):
                    self.samples.append((path, label))

        if len(self.samples) == 0:
            raise ValueError(f"No images found under {root}. "
                             f"Expected subfolders 'real' and 'fake'.")

        self.transform = transform or get_default_transform()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)
