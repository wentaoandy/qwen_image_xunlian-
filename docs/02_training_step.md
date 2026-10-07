# 02. One training step

This document describes one image-edit training step.

First, load one condition image, one target image, and one edit instruction.

Second, encode the condition image with Qwen3-VL together with the instruction.
This produces prompt embeddings and an image-slot mask.

Third, encode the same condition image with the VAE.
Use the clean latent as the condition latent.

Fourth, encode the target image with the VAE.
This produces the clean target latent.

Fifth, sample one random flow-matching timestep for the target.
Generate random Gaussian noise.
Interpolate the clean target latent and the noise with the sampled sigma.
Do not add this noise to the condition latent.

Sixth, pack the clean condition latent and the noisy target latent.
Put the condition first.
Put the target second.

Seventh, call the Transformer with the sampled target timestep.
The Transformer internally creates an extra `t=0` modulation row.
It applies this row to text and condition-image tokens.
It applies the sampled timestep row to target-image tokens.

Eighth, keep only the output tokens that belong to the target.
The condition prefix does not have a diffusion loss target.

Ninth, compute the flow-matching target:

`flow_target = noise - clean_target_latent`

Finally, compute the weighted mean squared error and backpropagate through the trainable LoRA layers.
