# Molmo Model Structure Printer

A tool for analyzing and visualizing the architecture of Molmo models.

## Overview

This script loads a Molmo model and provides detailed information about its structure, including:
- Total parameter count
- Module hierarchy
- Layer types and distributions
- Parameter groupings
- Configuration details

## Usage

### Basic Usage

```bash
# Analyze Molmo-7B-O model
python model_structure.py --model-path allenai/Molmo-7B-O-0924
```

### Quick Config Check (No Model Loading)

```bash
# Fast: only load config, don't load the full model
python model_structure.py --model-path allenai/Molmo-7B-O-0924 --no-load-model
```

### Detailed Analysis with Parameter Shapes

```bash
# Show parameter shapes (more verbose)
python model_structure.py --model-path allenai/Molmo-7B-O-0924 --show-shapes
```

### Custom Depth for Module Hierarchy

```bash
# Show deeper module hierarchy (default is 3)
python model_structure.py --model-path allenai/Molmo-7B-O-0924 --max-depth 5
```

## Available Models

- `allenai/Molmo-7B-D-0924` - Best 7B model for demos
- `allenai/Molmo-7B-O-0924` - Most open 7B model
- `allenai/MolmoE-1B-0924` - Efficient 1B MoE model
- `allenai/Molmo-72B-0924` - Best 72B model

## Output Sections

### 1. Model Summary
- Model class name
- Total parameters
- Trainable parameters

### 2. Model Config
Key configuration attributes like:
- Hidden size
- Number of layers
- Attention heads
- Vocabulary size
- Vision config

### 3. Module Hierarchy
Tree view of model modules up to specified depth:
```
├── vision_backbone: VisionBackbone (1.5B params)
│   ├── image_vit: ImageVit (1.5B params)
│   │   ├── embeddings: Embedding (20M params)
│   │   ├── encoder: Encoder (1.4B params)
├── model: LlamaModel (6.5B params)
│   ├── embed_tokens: Embedding (256M params)
│   ├── layers: ModuleList (5.8B params)
```

### 4. Top-Level Modules
Summary of main components with parameter counts and percentages:
```
vision_backbone                VisionBackbone                    1.5B (18.75%)
model                          LlamaModel                        6.5B (81.25%)
lm_head                        Linear                            256M (3.20%)
```

### 5. Parameter Details
Parameters grouped by component:
```
vision_backbone: 1.5B params
  vision_backbone.image_vit.embeddings.position_embedding
  vision_backbone.image_vit.encoder.layers.0.attention.q_proj.weight
  ...

model: 6.5B params
  model.embed_tokens.weight
  model.layers.0.self_attn.q_proj.weight
  ...
```

### 6. Layer Type Analysis
Count and parameters for each layer type:
```
Layer Type                               Count   Parameters
--------------------------------------------------------------------------------
Linear                                     512        5.2B
LayerNorm                                  256      512.0K
Embedding                                    2      256.5M
```

## Example Output

```
================================================================================
MOLMO MODEL STRUCTURE
================================================================================

Model Class: Molmo

Total Parameters: 7.85B (7,850,000,000)
Trainable Parameters: 7.85B (7,850,000,000)

--------------------------------------------------------------------------------
MODEL CONFIG
--------------------------------------------------------------------------------
  hidden_size: 4096
  num_hidden_layers: 32
  num_attention_heads: 32
  intermediate_size: 14336
  vocab_size: 152064
  max_position_embeddings: 8192
```

## Performance Notes

- **Loading time**: 30-60 seconds for 7B models (depending on hardware)
- **Memory usage**: ~15GB for 7B models (loads to CPU by default)
- **Use `--no-load-model`** for instant config checking without loading weights

## Use Cases

1. **Understanding Architecture**: See how the model is structured
2. **Debugging**: Find specific layers or modules
3. **Parameter Analysis**: Understand where parameters are allocated
4. **Model Comparison**: Compare different Molmo variants
5. **Development**: Reference when implementing custom models

## Tips

- Start with `--no-load-model` for quick config checks
- Use `--max-depth 2` for a cleaner overview
- Add `--show-shapes` when you need detailed parameter information
- The script loads models to CPU to avoid GPU memory issues

## Requirements

- transformers
- torch
- Model weights will be downloaded from Hugging Face (requires internet connection)

## See Also

- `server.py` - Run Molmo as a server
- `inference.py` - Standalone inference script
- `attention_map/patchify.py` - Visualize image processing
