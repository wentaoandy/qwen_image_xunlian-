import numpy as np
import torch


def pil_to_vae_tensor(image, resolution: int, device, dtype):
    image = image.resize((resolution, resolution))
    arr = np.asarray(image).astype("float32") / 255.0
    arr = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    arr = arr * 2.0 - 1.0
    # Qwen-Image-2.1 VAE expects a temporal dimension.
    return arr.unsqueeze(2).to(device=device, dtype=dtype)


@torch.no_grad()
def encode_edit_latents(vae, condition_image, target_image, resolution: int, dtype):
    device = next(vae.parameters()).device
    cond_pixels = pil_to_vae_tensor(condition_image, resolution, device, vae.dtype)
    target_pixels = pil_to_vae_tensor(target_image, resolution, device, vae.dtype)

    # The target uses a sampled latent. The clean condition uses the distribution mode.
    target_latent = vae.encode(target_pixels).latent_dist.sample()
    condition_latent = vae.encode(cond_pixels).latent_dist.mode()

    mean = torch.tensor(vae.config.latents_mean, device=device, dtype=target_latent.dtype).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=device, dtype=target_latent.dtype).view(1, -1, 1, 1, 1)

    target_latent = ((target_latent - mean) / std).to(dtype=dtype)
    condition_latent = ((condition_latent - mean) / std).to(dtype=dtype)
    return condition_latent, target_latent
