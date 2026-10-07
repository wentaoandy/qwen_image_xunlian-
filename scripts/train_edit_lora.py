import argparse
from pathlib import Path
import torch
from diffusers import QwenImage21Pipeline
from peft import get_peft_model_state_dict

from qwen_image_learning.config import load_config
from qwen_image_learning.data.edit_dataset import EditPairDataset
from qwen_image_learning.models.loaders import load_training_components
from qwen_image_learning.models.lora import add_transformer_lora
from qwen_image_learning.training.engine import train_loop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/edit_lora.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    components = load_training_components(cfg.model.path, cfg.model.dtype, device)
    transformer = add_transformer_lora(
        components["transformer"],
        cfg.train.lora_rank,
        cfg.train.lora_alpha,
    )
    components["transformer"] = transformer

    dataset = EditPairDataset(cfg.train.manifest)
    trained = train_loop(cfg, components, dataset)

    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    QwenImage21Pipeline.save_lora_weights(
        str(output_dir),
        transformer_lora_layers=get_peft_model_state_dict(trained),
    )
    print(f"Saved LoRA to {output_dir}")


if __name__ == "__main__":
    main()
