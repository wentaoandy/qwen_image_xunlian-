# 04. 3D condition research path

Use a simple baseline before you change the Transformer.

Render the 3D asset into 2D condition images.
Useful conditions include depth, normal, silhouette, edge, and multi-view RGB images.

The first experiment should not add a new 3D encoder.
Use the existing image-condition interface.
This test answers one question: can the native image-condition path learn enough geometry?

A product-repair sample can contain these inputs:

- Wrong product image.
- Edit mask.
- Correct reference image.
- Depth render.
- Normal render.
- Target correct image.

Inference already supports several condition images.
Training is more difficult.
The official learning script is easiest with one condition image.
A multi-condition training extension must keep three items consistent:

1. Qwen3-VL image slots for every condition image.
2. VAE latent blocks for every condition image.
3. `img_shapes` and image-block metadata in the same order.

Do not concatenate depth and normal tensors into the target channels without a clear reason.
That changes the input contract of `img_in`.
Start with separate condition blocks.

The second research stage can use a direct 3D encoder.
The encoder can output tokens with shape `[B, N3D, 4096]`.
A projection layer can map another encoder dimension to 4096.

A direct 3D-token design must solve three new problems:

- Define how 3D tokens enter the joint sequence.
- Define position encoding for 3D coordinates.
- Define attention permissions between 3D blocks and image blocks.

Do not reuse the existing `(frame, height, width)` RoPE as `(x, y, z)` without an explicit design.
Those axes have different semantics.

Keep one property if possible: the 3D condition block must not read the noisy target block.
Then the 3D condition can remain a stable prefix during inference.
This property keeps prefix KV caching possible.
