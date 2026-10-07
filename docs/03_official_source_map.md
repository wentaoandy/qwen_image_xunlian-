# 03. Official source map

Use this file when you want to compare this learning repository with the official implementation.

## Official image-edit LoRA training

File:

https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora_qwenimage21_img2img.py

Read these parts first:

1. Model and scheduler loading.
2. `transformer.requires_grad_(False)` and LoRA adapter insertion.
3. Condition-image prompt embedding cache.
4. VAE encoding for condition and target images.
5. Random timestep sampling.
6. Flow-matching noise interpolation.
7. Condition and target latent packing.
8. Transformer call.
9. Target-tail selection.
10. Flow target and loss.

## Official Transformer

File:

https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_qwenimage21.py

Read these parts first:

1. `QwenImage21Transformer2DModel.forward`.
2. `build_token_metadata`.
3. `target_token_mask`.
4. `causal_condition`.
5. `_select_modulation_rows`.
6. `QwenImage21TransformerBlock`.
7. `QwenImage21Attention`.
8. KV-cache classes.

The key code path is the timestep split.
The Transformer appends one zero timestep.
The target mask selects the sampled timestep for target tokens.
All non-target tokens use the zero timestep.

## Official inference pipeline

File:

https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py

Read these parts first:

1. Condition-image preprocessing.
2. Prompt encoding.
3. Condition-image VAE encoding.
4. `prepare_latents`.
5. Prefix KV-cache creation.
6. `extract` mode on the first denoising step.
7. `cached` mode on later denoising steps.
