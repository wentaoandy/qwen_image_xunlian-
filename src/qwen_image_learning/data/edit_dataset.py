import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class EditPairDataset(Dataset):
    """Load one edit condition, one target image, and one instruction."""

    def __init__(self, manifest: str | Path):
        self.manifest = Path(manifest)
        self.root = self.manifest.parent
        with self.manifest.open("r", encoding="utf-8") as f:
            self.items = [json.loads(line) for line in f if line.strip()]

    def __len__(self):
        return len(self.items)

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else (self.root.parent / p).resolve()

    def __getitem__(self, index: int):
        item = self.items[index]
        condition = Image.open(self._resolve(item["condition_image"])).convert("RGBA")
        target = Image.open(self._resolve(item["target_image"])).convert("RGBA")
        return {
            "condition_image": condition,
            "target_image": target,
            "prompt": item["prompt"],
        }
