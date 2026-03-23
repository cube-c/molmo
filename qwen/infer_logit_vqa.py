import argparse
import csv
import json
import os

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# Default HF model ID
_HF_DEFAULT = "Qwen/Qwen2.5-VL-3B-Instruct"


def load_model_and_processor(model_path: str):
    print(f"Loading Qwen2-VL model from {model_path}...")
    # For fine-tuned checkpoints, load the processor from the base model
    processor = AutoProcessor.from_pretrained(_HF_DEFAULT, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    return model, processor


@torch.inference_mode()
def get_yes_no_logits(model, processor, image_path, text):
    """Run a forward pass and return logits for 'Yes'/'No' tokens plus the real generated response."""
    tokenizer = processor.tokenizer

    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("No", add_special_tokens=False)[0]

    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ],
        }
    ]

    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[chat_text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # --- logits ---
    outputs = model(**inputs)
    last_logits = outputs.logits[0, -1, :]

    # --- greedy generation for real response ---
    # input_len = inputs["input_ids"].shape[1]
    # gen_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    # response = tokenizer.decode(gen_ids[0, input_len:], skip_special_tokens=True).strip()

    return last_logits[yes_id].item(), last_logits[no_id].item(), "" 


VARIANTS = {
    "obj1_closer": ("Is the {a} closer to the viewer than the {b}?", "Yes", "No"),
    "obj2_closer": ("Is the {b} closer to the viewer than the {a}?", "No", "Yes"),
    "obj1_farther": ("Is the {a} farther from the viewer than the {b}?", "No", "Yes"),
    "obj2_farther": ("Is the {b} farther from the viewer than the {a}?", "Yes", "No"),
}


def build_yes_no_question(entry, variant="obj1_closer"):
    """Build a yes/no question from a VQA entry.

    Returns (question_text, ground_truth) where ground_truth is 'Yes' or 'No'.
    """
    obj1 = entry["obj1"]
    obj2 = entry["obj2"]
    template, gt_closer, gt_farther = VARIANTS[variant]
    a = f"{obj1['color']} {obj1['shape']}"
    b = f"{obj2['color']} {obj2['shape']}"
    question = template.format(a=a, b=b) + " Answer with yes or no."
    gt = gt_closer if entry["answer"] == "closer" else gt_farther
    return question, gt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", "-m", type=str, default=_HF_DEFAULT,
                        help="HF model ID or local path to Qwen2-VL checkpoint")
    parser.add_argument("--vqa-json", type=str, required=True,
                        help="Path to vqa.json file")
    parser.add_argument("--image-root", type=str, default="/app/blender",
                        help="Root directory for resolving image paths in vqa.json")
    parser.add_argument("--output-csv", "-o", type=str, default="logit_results_vqa.csv")
    parser.add_argument("--variant", type=str, default=None,
                        choices=list(VARIANTS.keys()),
                        help="Question variant. If not specified, iterate over all variants.")
    args = parser.parse_args()

    model, processor = load_model_and_processor(args.model_path)

    with open(args.vqa_json) as f:
        vqa_data = json.load(f)

    variants = [args.variant] if args.variant else list(VARIANTS.keys())

    for variant in variants:
        if args.variant:
            out_csv = args.output_csv
        else:
            base, ext = os.path.splitext(args.output_csv)
            out_csv = f"{base}_{variant}{ext}"

        print(f"\n{'='*60}")
        print(f"Variant: {variant}  |  {VARIANTS[variant][0]}")
        print(f"{'='*60}")
        print(f"Loaded {len(vqa_data)} VQA entries from {args.vqa_json}")

        results = []
        correct = 0
        for i, entry in enumerate(vqa_data):
            image_path = os.path.join(args.image_root, entry["image"])
            question, gt = build_yes_no_question(entry, variant)

            yes_logit, no_logit, response = get_yes_no_logits(model, processor, image_path, question)
            pred = "Yes" if yes_logit > no_logit else "No"
            is_correct = pred == gt

            if is_correct:
                correct += 1

            results.append({
                "model_path": args.model_path,
                "variant": variant,
                "image": entry["image"],
                "question": question,
                "ground_truth": gt,
                "prediction": pred,
                "response": response,
                "Yes_logit": yes_logit,
                "No_logit": no_logit,
                "correct": is_correct,
            })
            print(f"[{i+1}/{len(vqa_data)}] {os.path.basename(entry['image'])}: "
                  f"Yes={yes_logit:.4f}, No={no_logit:.4f} | pred={pred} gt={gt} "
                  f"resp=\"{response}\" {'OK' if is_correct else 'WRONG'}")

        acc = correct / len(vqa_data) * 100 if vqa_data else 0
        print(f"\nAccuracy ({variant}): {correct}/{len(vqa_data)} ({acc:.1f}%)")

        fieldnames = ["model_path", "variant", "image", "question", "ground_truth", "prediction",
                      "response", "Yes_logit", "No_logit", "correct"]
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"Results saved to {out_csv}")


if __name__ == "__main__":
    main()
