#!/usr/bin/env python3
"""
Molmo Model Structure Printer

This script loads a Molmo model and prints its architecture in detail,
including layer names, types, and parameter counts.

Usage: python model_structure.py --model-path allenai/Molmo-7B-O-0924
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from collections import OrderedDict


def count_parameters(model):
    """Count total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def format_number(num):
    """Format large numbers with K, M, B suffixes."""
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    else:
        return str(num)


def print_module_structure(module, indent=0, max_depth=None, current_depth=0):
    """Recursively print module structure."""
    if max_depth is not None and current_depth >= max_depth:
        return

    prefix = "  " * indent

    for name, child in module.named_children():
        # Count parameters in this module
        params = sum(p.numel() for p in child.parameters())

        # Get module type
        module_type = child.__class__.__name__

        # Print module info
        if params > 0:
            print(f"{prefix}├── {name}: {module_type} ({format_number(params)} params)")
        else:
            print(f"{prefix}├── {name}: {module_type}")

        # Recursively print children
        if len(list(child.children())) > 0:
            print_module_structure(child, indent + 1, max_depth, current_depth + 1)


def print_detailed_structure(model):
    """Print detailed model structure."""
    print("\n" + "="*80)
    print("MOLMO MODEL STRUCTURE")
    print("="*80)

    # Print model class
    print(f"\nModel Class: {model.__class__.__name__}")

    # Print total parameters
    total_params, trainable_params = count_parameters(model)
    print(f"\nTotal Parameters: {format_number(total_params)} ({total_params:,})")
    print(f"Trainable Parameters: {format_number(trainable_params)} ({trainable_params:,})")

    # Print model config if available
    if hasattr(model, 'config'):
        print("\n" + "-"*80)
        print("MODEL CONFIG")
        print("-"*80)
        config = model.config

        # Print key config attributes
        important_attrs = [
            'hidden_size', 'num_hidden_layers', 'num_attention_heads',
            'num_key_value_heads', 'intermediate_size', 'vocab_size',
            'max_position_embeddings', 'model_type', 'vision_config'
        ]

        for attr in important_attrs:
            if hasattr(config, attr):
                value = getattr(config, attr)
                print(f"  {attr}: {value}")

    print("\n" + "-"*80)
    print("MODULE HIERARCHY")
    print("-"*80)
    print_module_structure(model, max_depth=3)

    print("\n" + "-"*80)
    print("TOP-LEVEL MODULES")
    print("-"*80)

    # Print top-level modules with parameter counts
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        param_str = format_number(params)
        pct = (params / total_params * 100) if total_params > 0 else 0
        print(f"  {name:30s} {module.__class__.__name__:30s} {param_str:>10s} ({pct:5.2f}%)")


def print_parameter_details(model, show_shapes=False):
    """Print detailed parameter information."""
    print("\n" + "-"*80)
    print("PARAMETER DETAILS")
    print("-"*80)

    param_groups = {}

    for name, param in model.named_parameters():
        # Group by first part of name (e.g., 'vision_backbone', 'model', etc.)
        group = name.split('.')[0]
        if group not in param_groups:
            param_groups[group] = []
        param_groups[group].append((name, param))

    # Print by group
    for group_name, params in sorted(param_groups.items()):
        group_params = sum(p.numel() for _, p in params)
        print(f"\n{group_name}: {format_number(group_params)} params")

        if show_shapes:
            for name, param in params[:5]:  # Show first 5 params in each group
                print(f"  {name:60s} {str(tuple(param.shape)):20s} {format_number(param.numel()):>10s}")
            if len(params) > 5:
                print(f"  ... ({len(params) - 5} more parameters in this group)")


def print_layer_analysis(model):
    """Analyze and print layer types."""
    print("\n" + "-"*80)
    print("LAYER TYPE ANALYSIS")
    print("-"*80)

    layer_counts = {}
    layer_params = {}

    for name, module in model.named_modules():
        module_type = module.__class__.__name__

        if module_type not in layer_counts:
            layer_counts[module_type] = 0
            layer_params[module_type] = 0

        layer_counts[module_type] += 1
        layer_params[module_type] += sum(p.numel() for p in module.parameters(recurse=False))

    # Sort by parameter count
    sorted_layers = sorted(layer_params.items(), key=lambda x: x[1], reverse=True)

    print(f"{'Layer Type':<40s} {'Count':>8s} {'Parameters':>15s}")
    print("-" * 80)

    for layer_type, params in sorted_layers:
        if params > 0:  # Only show layers with parameters
            count = layer_counts[layer_type]
            print(f"{layer_type:<40s} {count:>8d} {format_number(params):>15s}")


def main():
    parser = argparse.ArgumentParser(description="Molmo Model Structure Printer")
    parser.add_argument("--model-path", default="allenai/Molmo-7B-O-0924",
                       help="Model path (default: allenai/Molmo-7B-O-0924)")
    parser.add_argument("--show-shapes", action="store_true",
                       help="Show parameter shapes (can be verbose)")
    parser.add_argument("--max-depth", type=int, default=3,
                       help="Maximum depth for module hierarchy (default: 3)")
    parser.add_argument("--no-load-model", action="store_true",
                       help="Only load config, don't load full model (faster)")

    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")

    if args.no_load_model:
        # Only load config for faster analysis
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        print("\n" + "="*80)
        print("MODEL CONFIG (model not loaded, use without --no-load-model for full analysis)")
        print("="*80)
        print(config)
        return

    # Load full model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='cpu'  # Keep on CPU to avoid OOM
    )

    print("Model loaded successfully!\n")

    # Print detailed structure
    print_detailed_structure(model)

    # Print parameter details
    print_parameter_details(model, show_shapes=args.show_shapes)

    # Print layer analysis
    print_layer_analysis(model)

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
