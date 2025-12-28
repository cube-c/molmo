#!/usr/bin/env python3
"""
Standalone Molmo inference script.
Usage: python inference.py --image <path> --prompt "<text>"
"""

import argparse
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig


def main():
    parser = argparse.ArgumentParser(description="Molmo Standalone Inference")
    parser.add_argument("--image", required=True, help="Path to the image file")
    parser.add_argument("--prompt", required=True, help="Text prompt for the model")
    parser.add_argument("--model-path", default="allenai/Molmo-7B-D-0924", help="Model path (default: allenai/Molmo-7B-D-0924)")
    parser.add_argument("--max-tokens", type=int, default=200, help="Maximum tokens to generate (default: 200)")

    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")

    # Load processor
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )

    # Convert to bfloat16 for efficiency if CUDA is available
    if torch.cuda.is_available():
        model = model.to(dtype=torch.bfloat16)

    print(f"Model loaded successfully.")

    # Load image
    image = Image.open(args.image).convert("RGB")

    # Process inputs
    inputs = processor.process(
        images=[image],
        text=args.prompt
    )

    # Move inputs to device and create batch
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    # Ensure images are in bfloat16 if model is
    if "images" in inputs and model.dtype == torch.bfloat16:
        inputs["images"] = inputs["images"].to(torch.bfloat16)

    print(f"Generating response...")

    # Generate response
    with torch.inference_mode():
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=args.max_tokens, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer
        )

    # Decode generated tokens
    generated_tokens = output[0, inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Print result
    print("\nResponse:")
    print(f"Assistant: {generated_text}")


if __name__ == "__main__":
    main()
