import torch


@torch.no_grad()
def encode_condition_text_and_image(encoding_pipeline, prompt: str, condition_image, device):
    """Encode the instruction together with the condition image.

    Qwen-Image-2.1 uses the condition image twice.
    This function implements the Qwen3-VL path.
    The VAE path is implemented in training/latents.py.
    """
    prompt_embeds, prompt_mask, image_pad_mask = encoding_pipeline.encode_prompt(
        image=[condition_image],
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
    )
    return {
        "prompt_embeds": prompt_embeds,
        "prompt_mask": prompt_mask,
        "image_pad_mask": image_pad_mask,
    }
