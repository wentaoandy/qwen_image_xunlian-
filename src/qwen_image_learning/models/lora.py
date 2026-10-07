from peft import LoraConfig


def add_transformer_lora(transformer, rank: int, alpha: int):
    """Add LoRA to the same attention projections used by the official example."""
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    transformer.add_adapter(config)
    return transformer
