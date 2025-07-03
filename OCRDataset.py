from torch.utils.data import Dataset
from PIL import Image
import torch

from utils import text_to_labels

class OCRDataset(Dataset):
    def __init__(self, samples, transform=None):
        """
        samples: list of (image_path, label_str)
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")  # grayscale
        if self.transform:
            img = self.transform(img)

        # Convert label string -> list of ints
        label_encoded = torch.tensor(text_to_labels(label), dtype=torch.long)

        return img, label_encoded, len(label_encoded)
