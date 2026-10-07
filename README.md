# Qwen-Image-2.1 Learning Repository

This repository is a learning project for Qwen-Image-2.1 image-edit training.
It does not copy the full official Diffusers training script.
It separates the same core logic into small components so that each file has one job.

## What this repository teaches

1. Load the Qwen-Image-2.1 components.
2. Encode the edit condition in two paths.
3. Encode the target image with the VAE.
4. Sample one random flow-matching timestep for the target.
5. Keep the condition latent clean.
6. Put condition tokens before target tokens.
7. Train LoRA on the Transformer.
8. Extend the condition path for depth, normal, and other 3D renders.

## Project structure

```text
qwen_image_xunlian-/
├── configs/
│   └── edit_lora.yaml
├── data/
│   └── example_manifest.jsonl
├── docs/
│   ├── 01_architecture.md
│   ├── 02_training_step.md
│   ├── 03_official_source_map.md
│   └── 04_3d_condition.md
├── scripts/
│   ├── train_edit_lora.py
│   └── infer_edit.py
├── src/qwen_image_learning/
│   ├── config.py
│   ├── data/
│   │   └── edit_dataset.py
│   ├── models/
│   │   ├── loaders.py
│   │   ├── conditioning.py
│   │   └── lora.py
│   ├── training/
│   │   ├── latents.py
│   │   ├── noise.py
│   │   ├── step.py
│   │   └── engine.py
│   └── conditions/
│       └── three_d.py
├── requirements.txt
└── pyproject.toml
```

## Why the code is split this way

`data/` defines what one training sample contains.

`models/` defines model loading, condition encoding, and LoRA insertion.

`training/` defines the diffusion training mechanics. It does not know where the files came from.

`conditions/` defines optional condition types. The first extension is 3D-rendered conditions.

`scripts/` is the entry layer. A script connects the components. A script must not contain all model logic.

`docs/` explains the design before you change the code.

## Official source

The learning code follows the current Hugging Face Diffusers implementation of Qwen-Image-2.1.

Official image-edit LoRA training:
https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora_qwenimage21_img2img.py

Transformer implementation:
https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_qwenimage21.py

Inference pipeline:
https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py

## Important scope

The first runnable training path uses one condition image and one target image.
This matches the easiest official edit-training path.

Qwen-Image-2.1 can use multiple condition images during inference.
Multi-condition training needs extra work for prompt image slots, image block shapes, and masks.
The 3D document explains this extension.
