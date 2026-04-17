#!/usr/bin/env python3
"""
Molmo Image Patchification Visualizer

This script processes images through Molmo's processor and saves the individual
patches/crops that are fed to the vision encoder.

Usage: python patchify.py --image <path> --output-dir <dir>
"""

import argparse
import os
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# OpenAI CLIP normalization constants
OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

def denormalize_openai_clip(x: torch.Tensor) -> torch.Tensor:
    """
    Denormalize an image tensor normalized with OpenAI CLIP mean/std.

    Accepts:
      - x shape [3, H, W]  or
      - x shape [H, W, 3]  or
      - x shape [..., 3, H, W] (batched)

    Returns:
      - tensor in [0, 1] (clipped)
    """
    mean = torch.tensor(OPENAI_CLIP_MEAN, device=x.device, dtype=x.dtype)
    std  = torch.tensor(OPENAI_CLIP_STD,  device=x.device, dtype=x.dtype)

    if x.ndim >= 3 and x.shape[-3] == 3:
        # [..., 3, H, W]
        mean = mean.view(*([1] * (x.ndim - 3)), 3, 1, 1)
        std  = std.view(*([1] * (x.ndim - 3)), 3, 1, 1)
        y = x * std + mean
    elif x.ndim == 3 and x.shape[-1] == 3:
        # [H, W, 3]
        y = x * std.view(1, 1, 3) + mean.view(1, 1, 3)
    else:
        raise ValueError(f"Unexpected shape {tuple(x.shape)}. Expected CHW, HWC, or batched CHW.")

    return y.clamp(0, 1)

def patches_to_crop(patches_576x588: torch.Tensor):
    # (576, 588) -> (24, 24, 14, 14, 3) -> (336, 336, 3)
    x = patches_576x588.reshape(24, 24, 14, 14, 3)
    x = x.permute(0, 2, 1, 3, 4).reshape(24*14, 24*14, 3)  # (336,336,3)
    x = denormalize_openai_clip(x)
    return x

def save_patches(image_path: str, output_dir: str, model_path: str = "allenai/Molmo-7B-D-0924"):
    """Process an image and save all patches created by Molmo's processor."""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading processor from {model_path}...")
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype='auto'
    )

    # Load image
    print(f"Loading image from {image_path}...")
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    print(f"Original image size: {original_size}")

    # Process image
    print("Processing image through Molmo processor...")
    inputs = processor.process(
        images=[image],
        text="" 
    )

    # Extract image data
    if "images" in inputs:
        images_tensor = inputs["images"]
        print(f"Processed images tensor shape: {images_tensor.shape}")

        # The shape is typically [num_crops, channels, height, width]
        num_crops = images_tensor.shape[0]
        print(f"Number of crops/patches: {num_crops}")

        # Save individual patches
        for i in range(num_crops):
            patch = images_tensor[i]
            patches_to_crop_tensor = patches_to_crop(patch).cpu().numpy()
            print(f"Patch {i} shape after crop conversion: {patches_to_crop_tensor.shape}")
            print("Min/Max values:", patches_to_crop_tensor.min().item(), patches_to_crop_tensor.max().item())

            # Convert to uint8
            patch_np = (patches_to_crop_tensor * 255).astype(np.uint8)

            # Create PIL Image
            patch_img = Image.fromarray(patch_np)

            # Save patch
            patch_filename = os.path.join(output_dir, f"patch_{i:02d}.png")
            patch_img.save(patch_filename)
            print(f"Saved patch {i} to {patch_filename} (size: {patch_img.size})")

    # Create a visualization showing all patches
    if num_crops > 0:
        create_patch_grid(output_dir, num_crops, original_size, image_path)

    print(f"\nAll patches saved to {output_dir}/")


def create_patch_grid(output_dir: str, num_patches: int, original_size: tuple, original_image_path: str):
    """Create a grid visualization of all patches."""

    # Load all patches
    patch_images = []
    for i in range(num_patches):
        patch_path = os.path.join(output_dir, f"patch_{i:02d}.png")
        if os.path.exists(patch_path):
            patch_images.append(Image.open(patch_path))

    if not patch_images:
        return

    # Create grid layout
    cols = min(4, num_patches)
    rows = (num_patches + cols - 1) // cols

    # Get patch size (assuming all patches are same size)
    patch_size = patch_images[0].size

    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)

    # Plot each patch
    for i, (ax, patch_img) in enumerate(zip(axes.flat, patch_images)):
        ax.imshow(patch_img)
        ax.set_title(f"Patch {i}", fontsize=10)
        ax.axis('off')

    # Hide empty subplots
    for i in range(num_patches, rows * cols):
        axes.flat[i].axis('off')

    plt.suptitle(f"Molmo Image Patches ({num_patches} patches, {patch_size[0]}x{patch_size[1]} each)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save grid
    grid_path = os.path.join(output_dir, "patch_grid.png")
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    print(f"Saved patch grid visualization to {grid_path}")
    plt.close()

    # Also create visualization with patches
    create_visualization(output_dir, num_patches, patch_size)


def create_visualization(output_dir: str, num_patches: int, patch_size: tuple):
    """Create a visualization of patches with red borders."""

    # Load all patches (skip first one as per user's modification)
    patch_images = []
    for i in range(num_patches):
        patch_path = os.path.join(output_dir, f"patch_{i:02d}.png")
        if os.path.exists(patch_path):
            patch_images.append(Image.open(patch_path))

    if not patch_images:
        return

    # Create a montage with red borders
    cols = min(4, len(patch_images))
    rows = (len(patch_images) + cols - 1) // cols

    border_width = 3  # Red border width in pixels
    montage_width = patch_size[0] * cols
    montage_height = patch_size[1] * rows
    montage = Image.new('RGB', (montage_width, montage_height), color=(255, 255, 255))

    # Draw for adding red borders
    draw = ImageDraw.Draw(montage)

    # Paste patches and draw red borders (skip first patch as per user's modification)
    for idx, patch_img in enumerate(patch_images[1:]):
        row = idx // cols
        col = idx % cols
        x = col * patch_size[0]
        y = row * patch_size[1]

        # Paste the patch
        montage.paste(patch_img, (x, y))

        # Draw red border around the patch
        draw.rectangle(
            [x, y, x + patch_size[0] - 1, y + patch_size[1] - 1],
            outline=(255, 0, 0),
            width=border_width
        )

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    ax.imshow(montage)
    ax.set_title(f"Molmo Processed Patches\n{num_patches - 1} patches of {patch_size[0]}x{patch_size[1]} (excluding global patch)",
                 fontsize=12, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()

    visualization_path = os.path.join(output_dir, "visualization.png")
    plt.savefig(visualization_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {visualization_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Molmo Image Patchification Visualizer")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output-dir", default="./patch_output", help="Output directory for patches (default: ./patch_output)")
    parser.add_argument("--model-path", default="allenai/Molmo-7B-O-0924", help="Molmo model path (default: allenai/Molmo-7B-O-0924)")

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return

    # Process and save patches
    save_patches(args.image, args.output_dir, args.model_path)

    print("\n" + "="*60)
    print("Processing complete!")
    print(f"Output directory: {args.output_dir}")
    print("Files generated:")
    print("  - patch_XX.png: Individual patches")
    print("  - patch_grid.png: Grid visualization of all patches")
    print("  - visualization.png: Patches montage with red borders")
    print("="*60)


if __name__ == "__main__":
    main()
