import argparse
import torch
from PIL import Image
from diffusers import QwenImage21Pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", default="result.png")
    parser.add_argument("--lora", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pipe = QwenImage21Pipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda")

    if args.lora:
        pipe.load_lora_weights(args.lora)

    condition = Image.open(args.image).convert("RGBA")
    result = pipe(
        prompt=args.prompt,
        image=condition,
        num_inference_steps=40,
        generator=torch.Generator("cuda").manual_seed(args.seed),
    ).images[0]
    result.save(args.output)


if __name__ == "__main__":
    main()
