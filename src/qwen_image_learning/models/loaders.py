import torch
from diffusers import (
    AutoencoderKLQwenImage21,
    FlowMatchEulerDiscreteScheduler,
    QwenImage21Pipeline,
    QwenImage21Transformer2DModel,
)
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor


def resolve_dtype(name: str):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def load_training_components(model_path: str, dtype_name: str, device: torch.device):
    """Load each Qwen-Image-2.1 component separately.

    The split is intentional. Training code must know which component is frozen and which component is trainable.
    """
    dtype = resolve_dtype(dtype_name)

    processor = Qwen3VLProcessor.from_pretrained(model_path, subfolder="processor")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
    vae = AutoencoderKLQwenImage21.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype)
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        subfolder="text_encoder",
        torch_dtype=dtype,
    )
    transformer = QwenImage21Transformer2DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=dtype,
    )

    vae.requires_grad_(False).eval().to(device)
    text_encoder.requires_grad_(False).eval().to(device)
    transformer.requires_grad_(False).to(device)

    # This small pipeline is used only for Qwen3-VL prompt and condition-image encoding.
    encoding_pipeline = QwenImage21Pipeline.from_pretrained(
        model_path,
        vae=None,
        transformer=None,
        scheduler=None,
        processor=processor,
        text_encoder=text_encoder,
        torch_dtype=dtype,
    )

    return {
        "dtype": dtype,
        "processor": processor,
        "scheduler": scheduler,
        "vae": vae,
        "text_encoder": text_encoder,
        "transformer": transformer,
        "encoding_pipeline": encoding_pipeline,
    }
