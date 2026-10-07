import torch
from diffusers.training_utils import compute_density_for_timestep_sampling


def sample_timesteps(scheduler, batch_size: int, device):
    """Sample random training timesteps for the target image only."""
    u = compute_density_for_timestep_sampling(
        weighting_scheme="none",
        batch_size=batch_size,
        logit_mean=0.0,
        logit_std=1.0,
        mode_scale=1.29,
    )
    indices = (u * scheduler.config.num_train_timesteps).long()
    return scheduler.timesteps[indices].to(device)


def get_sigmas(scheduler, timesteps, ndim: int, dtype):
    schedule_timesteps = scheduler.timesteps.to(timesteps.device)
    sigmas = scheduler.sigmas.to(device=timesteps.device, dtype=dtype)
    indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[indices].flatten()
    while sigma.ndim < ndim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def add_flow_matching_noise(clean_target, noise, sigma):
    """Flow matching interpolation used by the official training path."""
    return (1.0 - sigma) * clean_target + sigma * noise
