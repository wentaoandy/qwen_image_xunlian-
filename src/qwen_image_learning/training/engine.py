import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from qwen_image_learning.models.conditioning import encode_condition_text_and_image
from qwen_image_learning.training.latents import encode_edit_latents
from qwen_image_learning.training.step import qwen21_edit_training_step


def train_loop(cfg, components, dataset):
    accelerator = Accelerator(gradient_accumulation_steps=cfg.train.gradient_accumulation_steps)
    transformer = components["transformer"]
    vae = components["vae"]
    scheduler = components["scheduler"]
    encoding_pipeline = components["encoding_pipeline"]
    dtype = components["dtype"]

    params = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.train.learning_rate)

    # PIL objects are kept as a Python list. This is an educational batch-size-1 path.
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        collate_fn=lambda x: x,
    )

    transformer, optimizer, loader = accelerator.prepare(transformer, optimizer, loader)
    global_step = 0

    while global_step < cfg.train.max_steps:
        for batch_list in loader:
            sample = batch_list[0]
            with accelerator.accumulate(transformer):
                cond = encode_condition_text_and_image(
                    encoding_pipeline,
                    sample["prompt"],
                    sample["condition_image"],
                    accelerator.device,
                )

                condition_latent, target_latent = encode_edit_latents(
                    vae,
                    sample["condition_image"],
                    sample["target_image"],
                    cfg.train.resolution,
                    dtype,
                )

                loss, debug = qwen21_edit_training_step(
                    transformer,
                    scheduler,
                    condition_latent,
                    target_latent,
                    cond["prompt_embeds"],
                    cond["prompt_mask"],
                    cond["image_pad_mask"],
                )

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                accelerator.print(
                    f"step={global_step} loss={loss.item():.6f} "
                    f"t={debug['timesteps'].flatten()[0].item():.1f}"
                )

            if global_step >= cfg.train.max_steps:
                break

    return accelerator.unwrap_model(transformer)
