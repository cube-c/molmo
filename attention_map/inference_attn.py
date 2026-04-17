#!/usr/bin/env python3
"""
Standalone Molmo inference script.
Usage: python inference.py --image <path> --prompt "<text>"
"""

import argparse
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from torch import Tensor
from typing import Tuple


# def act_hook(module, in_values: Tuple[Tensor], out_values: Tuple[Tensor]) -> None:
    # # setattr(module, ACT_ATTR_NAME, out_values)
    # module.acts_list.append(out_values[1]) # (Tensor, DynamicCache)
    # # print("Registered act hook, value shape", out_values[1].shape)

# def add_act_hooks(model, layer_index):
    # print(f"Adding activations on model layer {layer_index}!")

    # # Hook into the LLM layer's self-attention module
    # # https://huggingface.co/allenai/Molmo-7B-O-0924/blob/main/modeling_molmo.py#L421
    # print("Model LLM layers:", model.model.transformer.blocks[layer_index])
    # llm_layer = model.model.transformer.blocks[layer_index]
    # setattr(llm_layer, "acts_list", [])
    # setattr(llm_layer, "time_list", [])
    # hook = llm_layer.register_forward_hook(act_hook)
    # return hook


store = {}

def attn_from_calc(model):
    


def hook_att_proj(model, layer_idx: int):
    block = model.model.transformer.blocks[layer_idx]
    fused_dims = block.fused_dims  # (q_dim, k_dim, v_dim)

    def _hook(mod, inp, out):
        qkv = out.detach()
        qkv = qkv.clone()

        q, k, v = qkv.split(fused_dims, dim=-1)
        store["q"] = q
        store["k"] = k
        store["v"] = v

    return block.att_proj.register_forward_hook(_hook)

def main():
    parser = argparse.ArgumentParser(description="Molmo Standalone Inference")
    parser.add_argument("--image", required=True, help="Path to the image file")
    parser.add_argument("--prompt", required=True, help="Text prompt for the model")
    parser.add_argument("--model-path", default="allenai/Molmo-7B-O-0924", help="Model path (default: allenai/Molmo-7B-O-0924)")
    parser.add_argument("--max-tokens", type=int, default=500, help="Maximum tokens to generate (default: 500)")

    args = parser.parse_args()
    device = "cuda:7" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {args.model_path} on device {device}...")

    # Load processor
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype='auto',
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype='auto',
        device_map=device
    )

    # Convert to bfloat16 for efficiency if CUDA is available
    # if torch.cuda.is_available():
        # model = model.to(dtype=torch.bfloat16)

    print(f"Model loaded successfully.")
    print(model.model.transformer.blocks[30])

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
    print("Generated Text:\n", generated_text)

    # Re-process text
    inputs = processor.process(
        images=[image],
        text=args.prompt + generated_text
    )
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
    hook_att_proj(model, layer_idx=30)
    
    with torch.no_grad():
        out = model(
            **inputs,             
            use_cache=False,       
            output_hidden_states=True,
            return_dict=True,
        )

    print(store["q"].shape, store["k"].shape, store["v"].shape)

if __name__ == "__main__":
    main()
