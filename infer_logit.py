import argparse
import csv
import glob
import inspect
import os

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

# HF base model used for the processor when loading native checkpoints
_HF_BASE = "allenai/Molmo-7B-O-0924"

# Keys accepted by the native Molmo.forward() signature
_NATIVE_FORWARD_KEYS = set(inspect.signature(
    # imported lazily; populated at load time
    object
).parameters) if False else None


def is_native_checkpoint(path: str) -> bool:
    """Return True if path looks like a native Molmo unsharded checkpoint."""
    return os.path.isfile(os.path.join(path, "model.pt"))


def load_model_and_processor(model_path: str):
    if is_native_checkpoint(model_path):
        print(f"Detected native checkpoint at {model_path}")
        import sys
        sys.path.insert(0, "/app/molmo")
        from olmo.model import Molmo

        # Collect valid forward kwargs once
        global _NATIVE_FORWARD_KEYS
        _NATIVE_FORWARD_KEYS = set(inspect.signature(Molmo.forward).parameters)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Molmo.from_checkpoint(model_path, device=device)
        model.eval()

        processor = AutoProcessor.from_pretrained(
            _HF_BASE,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
        )
        return model, processor, device
    else:
        print(f"Loading HF model from {model_path}...")
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
        )
        model.eval()
        return model, processor, None


@torch.inference_mode()
def get_yes_no_logits(model, processor, image_path, text, device=None):
    """Run a forward pass and return logits for 'Yes' and 'No' tokens."""
    tokenizer = processor.tokenizer

    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("No", add_special_tokens=False)[0]

    image = Image.open(image_path).convert("RGB")
    inputs = processor.process(images=[image], text=text)

    target = device if device else model.device
    inputs = {k: v.to(target).unsqueeze(0) for k, v in inputs.items()}

    # For native models, filter to only forward-compatible keys
    if _NATIVE_FORWARD_KEYS is not None:
        inputs = {k: v for k, v in inputs.items() if k in _NATIVE_FORWARD_KEYS}

    outputs = model(**inputs)
    last_logits = outputs.logits[0, -1, :]

    return last_logits[yes_id].item(), last_logits[no_id].item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", "-m", type=str, default=_HF_BASE,
                        help="HF model ID or path to a native unsharded checkpoint (contains model.pt)")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--media-dir", type=str, required=True,
                        help="Directory containing images (*.png)")
    parser.add_argument("--output-csv", "-o", type=str, default="logit_results.csv")
    args = parser.parse_args()

    model, processor, device = load_model_and_processor(args.model_path)

    image_paths = sorted(glob.glob(os.path.join(args.media_dir, "*.png")))
    if not image_paths:
        print(f"No images found in {args.media_dir}")
        return

    print(f"Found {len(image_paths)} images in {args.media_dir}")

    results = []
    for i, image_path in enumerate(image_paths):
        yes_logit, no_logit = get_yes_no_logits(model, processor, image_path, args.text, device)
        results.append({
            "model_path": args.model_path,
            "text": args.text,
            "image": os.path.join(os.path.basename(os.path.dirname(image_path)), os.path.basename(image_path)),
            "Yes_logit": yes_logit,
            "No_logit": no_logit,
        })
        print(f"[{i+1}/{len(image_paths)}] {os.path.basename(image_path)}: "
              f"Yes={yes_logit:.4f}, No={no_logit:.4f}")

    fieldnames = ["model_path", "text", "image", "Yes_logit", "No_logit"]
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {args.output_csv}")


if __name__ == "__main__":
    main()
