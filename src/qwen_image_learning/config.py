from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class ModelConfig:
    path: str
    dtype: str = "bf16"


@dataclass
class TrainConfig:
    manifest: str
    output_dir: str
    resolution: int = 1024
    batch_size: int = 1
    learning_rate: float = 1e-4
    max_steps: int = 1000
    gradient_accumulation_steps: int = 1
    seed: int = 42
    lora_rank: int = 16
    lora_alpha: int = 16
    num_workers: int = 0


@dataclass
class Config:
    model: ModelConfig
    train: TrainConfig


def load_config(path: str | Path) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(
        model=ModelConfig(**raw["model"]),
        train=TrainConfig(**raw["train"]),
    )
