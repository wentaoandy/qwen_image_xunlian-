# 01. Qwen-Image-2.1 architecture

Qwen-Image-2.1 separates input encoding from denoising.

The Qwen3-VL text encoder reads the instruction and the condition image.
It produces semantic context.

The VAE also reads the condition image.
It produces clean image latents.

The VAE reads the target image during training.
It produces the target latent.
The training code adds noise only to this target latent.

The Transformer receives a joint sequence.
Condition image latents come before target image latents.
Text and image context are part of the same Transformer sequence.

The model uses block-causal attention.
A later image block can read an earlier image block.
Tokens inside one image block can read each other.
An earlier condition block cannot read the later target block.

The Transformer uses `causal_condition=True` by default.
The Transformer gives text and condition-image tokens a timestep modulation from `t=0`.
It gives target-image tokens the sampled diffusion timestep.

This design makes the condition prefix independent of the denoising step.
Inference can cache the prefix K and V tensors.
Training still recomputes them because trainable LoRA weights change.
