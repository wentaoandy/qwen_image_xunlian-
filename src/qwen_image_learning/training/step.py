import torch
from diffusers import QwenImage21Pipeline
from diffusers.training_utils import compute_loss_weighting_for_sd3
from .noise import sample_timesteps, get_sigmas, add_flow_matching_noise


def qwen21_edit_training_step(
    transformer,
    scheduler,
    condition_latent,
    target_latent,
    prompt_embeds,
    prompt_mask,
    image_pad_mask,
    vae_scale_factor: int = 16,
):
    """Run one Qwen-Image-2.1 edit-training forward pass.

    Important rule:
    - The condition latent stays clean.
    - The target latent receives random flow-matching noise.
    - The Transformer assigns t=0 modulation to text and condition tokens internally when causal_condition=True.
    """
    batch_size = target_latent.shape[0]
    noise = torch.randn_like(target_latent)
    timesteps = sample_timesteps(scheduler, batch_size, target_latent.device)
    sigmas = get_sigmas(scheduler, timesteps, target_latent.ndim, target_latent.dtype)
    noisy_target = add_flow_matching_noise(target_latent, noise, sigmas)

    cond_h, cond_w = condition_latent.shape[-2:]
    target_h, target_w = target_latent.shape[-2:]

    packed_cond = QwenImage21Pipeline._pack_latents(
        condition_latent,
        batch_size=batch_size,
        num_channels_latents=condition_latent.shape[1],
        height=cond_h,
        width=cond_w,
    )
    packed_target = QwenImage21Pipeline._pack_latents(
        noisy_target,
        batch_size=batch_size,
        num_channels_latents=noisy_target.shape[1],
        height=target_h,
        width=target_w,
    )

    hidden_states = torch.cat([packed_cond, packed_target], dim=1)
    img_shapes = [[(1, cond_h, cond_w), (1, target_h, target_w)]] * batch_size

    # Qwen3-VL image slots represent 2x2 groups of image latent tokens.
    target_slots = (target_h * target_w) // 4
    target_mask = torch.ones(
        batch_size,
        target_slots,
        dtype=image_pad_mask.dtype,
        device=image_pad_mask.device,
    )
    img_mask = torch.cat([image_pad_mask, target_mask], dim=1)

    pred = transformer(
        hidden_states=hidden_states,
        encoder_hidden_states=prompt_embeds,
        encoder_hidden_states_mask=prompt_mask,
        timestep=timesteps / 1000,
        img_shapes=img_shapes,
        img_mask=img_mask,
        return_dict=False,
    )[0]

    # Keep only the target tail. The condition prefix has no training target.
    pred = pred[:, -packed_target.shape[1]:]
    pred = QwenImage21Pipeline._unpack_latents(
        pred,
        target_h * vae_scale_factor,
        target_w * vae_scale_factor,
        vae_scale_factor,
    )

    flow_target = noise - target_latent
    weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)
    loss = ((pred.float() - flow_target.float()) ** 2 * weighting.float()).reshape(batch_size, -1).mean()

    return loss, {
        "timesteps": timesteps.detach(),
        "sigma": sigmas.detach(),
    }
