# Molmo Image Patchification Visualizer

This directory contains tools for visualizing how Molmo processes images internally by splitting them into patches/crops.

## Overview

Molmo uses a multi-crop strategy to process images at different scales. The `patchify.py` script extracts and visualizes these individual patches.

## Files

- `patchify.py` - Main script to extract and save image patches
- This README

## Usage

### Basic Usage

```bash
python patchify.py --image /path/to/image.jpg --output-dir ./patches
```

### With Custom Model

```bash
python patchify.py \
  --image /path/to/image.jpg \
  --output-dir ./patches \
  --model-path allenai/Molmo-7B-D-0924
```

## Output

The script generates:

1. **Individual Patches** (`patch_00.png`, `patch_01.png`, etc.)
   - Each patch that Molmo's vision encoder processes
   - Typically resized to a fixed size (e.g., 336x336)
   - Patch 0 is the global patch (entire image)
   - Patches 1+ are local crops for finer details

2. **Patch Grid** (`patch_grid.png`)
   - All patches arranged in a grid for easy visualization
   - Shows the total number of patches and their dimensions

3. **Visualization** (`visualization.png`)
   - Montage of local patches (excluding global patch)
   - Each patch has a red border for clear separation
   - Arranged in a 3-column grid

## Understanding the Output

Molmo processes images using multiple crops/patches to capture details at different scales:

- **Global patch**: The entire image resized to fit the vision encoder
- **Local patches**: Overlapping crops of the image for finer details
- **Crop mode**: Typically "overlap-and-resize-c2" with max 12 crops

The number and arrangement of patches depends on:
- Original image size
- Model configuration (max_crops, crop_mode)
- Aspect ratio

## Example

```bash
# Process an image
python patchify.py --image demo_image.jpg --output-dir ./my_patches

# Output will be in ./my_patches/:
# - patch_00.png (global patch)
# - patch_01.png, ..., patch_11.png (local patches)
# - patch_grid.png (all patches in grid)
# - visualization.png (local patches with red borders)
```

## Requirements

- transformers
- torch
- PIL (Pillow)
- matplotlib
- numpy

## Technical Details

The script:
1. Loads the Molmo processor
2. Processes an image through `processor.process()`
3. Extracts the `images` tensor from the processed inputs
4. Denormalizes and saves each patch
5. Creates visualizations for easy inspection

This is useful for:
- Understanding how Molmo sees your images
- Debugging vision-related issues
- Analyzing the multi-scale processing strategy
- Visualizing the model's input preprocessing
