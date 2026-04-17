import argparse
import importlib.util
import json
import os
import re
import torch

import cv2
from pydantic import BaseModel
from termcolor import colored
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import llava
from llava import conversation as clib
from llava.media import Image
from llava.model.configuration_llava import JsonSchemaResponseFormat, ResponseFormat

import time
from typing import Tuple
from torch import Tensor

def act_hook(module, in_values: Tuple[Tensor], out_values: Tuple[Tensor]) -> None:
    # setattr(module, ACT_ATTR_NAME, out_values)
    module.acts_list.append(out_values[1]) # (Tensor, DynamicCache)
    # print("Registered act hook, value shape", out_values[1].shape)

def add_act_hooks(model, layer_index):
    print(f"Adding activations on model layer {layer_index}!")
    print("Model LLM layers:", model.llm.model.layers[layer_index].self_attn)
    llm_layer = model.llm.model.layers[layer_index].self_attn
    setattr(llm_layer, "acts_list", [])
    setattr(llm_layer, "time_list", [])
    hook = llm_layer.register_forward_hook(act_hook)
    return hook

def remove_act_hooks(model, layer_index, hook):
    """Remove hooks and clear activation lists."""
    hook.remove()
    llm_layer = model.llm.model.layers[layer_index].self_attn
    if hasattr(llm_layer, "acts_list"):
        delattr(llm_layer, "acts_list")
    if hasattr(llm_layer, "time_list"):
        delattr(llm_layer, "time_list")


def get_spatial_token_ranges(grid_size=11, tile_width=4, tile_height=3, tile_offset=0):
    """
    Calculate spatial token index ranges for all image patches.

    Each patch has: [start_token, spatial_tokens (grid_size x grid_size), end_token]
    Returns only the ranges of spatial tokens (excluding start/end tokens).

    Args:
        grid_size: Spatial grid size per tile (e.g., 11 for 11x11)
        tile_width: Number of tiles horizontally
        tile_height: Number of tiles vertically
        tile_offset: Offset to start from a specific tile (e.g., 12 to start from tile 12)

    Returns:
        list of tuples: [(start_idx, end_idx), ...] for each patch's spatial tokens
    """
    tokens_per_tile = 2 + grid_size * grid_size  # 123 tokens per tile
    num_tiles = tile_width * tile_height

    ranges = []
    for tile_idx in range(num_tiles):
        # Apply tile_offset to calculate global tile index
        global_tile_idx = tile_offset + tile_idx
        tile_start = global_tile_idx * tokens_per_tile
        tile_end = tile_start + tokens_per_tile

        # Spatial tokens: skip start (index 0) and end (index -1)
        spatial_start = tile_start + 1
        spatial_end = tile_end - 1

        ranges.append((spatial_start, spatial_end))

    return ranges


def analyze_token_positions(model, prompt):
    """Analyze and return information about token positions in the input."""
    from llava.utils.media import extract_media
    from llava.utils.tokenizer import tokenize_conversation
    from llava.constants import DEFAULT_IMAGE_TOKEN

    # Prepare conversation and tokenize
    conversation = [{"from": "human", "value": prompt}]
    media = extract_media(conversation, config=model.config)

    # Get tokenized input
    input_ids = tokenize_conversation(conversation, model.tokenizer, add_generation_prompt=True)

    # Find positions of image tokens
    image_token_id = model.tokenizer.media_token_ids.get("image", None)

    print(f"\n=== Input Token Structure ===")
    print(f"Total input tokens (before image embedding): {len(input_ids)}")
    print(f"Image token ID: {image_token_id}")

    # Find image token positions in input_ids
    image_token_positions = []
    text_token_positions = []
    for i, token_id in enumerate(input_ids):
        if token_id == image_token_id:
            image_token_positions.append(i)
        else:
            text_token_positions.append(i)

    print(f"Image token positions in input_ids: {image_token_positions}")
    print(f"Number of text tokens in input_ids: {len(text_token_positions)}")

    return {
        "input_ids": input_ids,
        "image_token_id": image_token_id,
        "image_token_positions": image_token_positions,
        "text_token_positions": text_token_positions,
        "total_input_tokens": len(input_ids)
    }


def visualize_image_attention_per_token(
    square_attention,
    image_embedding_size,
    base_seq_len,
    tokenizer,
    output_ids,
    output_path="image_attention_per_token.png",
    num_per_row=4,
    grid_size=11,
    tile_width=4,
    tile_height=3,
    tile_offset=0,
):
    """
    Visualize attention weights to image tokens for each generated token as spatial heatmaps.

    For multi-tile images, tiles are arranged horizontally first (row-major):
    [Tile(0,0), Tile(0,1), ..., Tile(1,0), Tile(1,1), ...]

    Each tile: [start, 121 spatial tokens (11x11), end] = 123 tokens
    Total: tile_width × tile_height × 123 tokens

    Args:
        square_attention: Full attention matrix [batch, heads, seq_len, seq_len]
        image_embedding_size: Number of image embedding tokens
        base_seq_len: Length of base sequence (image + prompt)
        response_text: Generated response text
        tokenizer: Tokenizer for decoding tokens
        output_ids: Generated token IDs
        output_path: Path to save visualization
        num_per_row: Number of plots per row
        grid_size: Spatial grid size per tile (default: 11)
        tile_width: Number of tiles horizontally
        tile_height: Number of tiles vertically
    """
    import numpy as np
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if image_embedding_size == 0:
        raise ValueError("No image tokens found")

    # Each tile: 1 start + grid_size^2 spatial + 1 end
    tokens_per_tile = 2 + grid_size * grid_size  # 123 tokens per tile
    expected_tokens = tile_width * tile_height * tokens_per_tile

    if image_embedding_size != expected_tokens:
        import warnings
        warnings.warn(
            f"Image embedding size mismatch: expected {expected_tokens} tokens "
            f"({tile_height}×{tile_width} tiles, {tokens_per_tile} each), "
            f"but got {image_embedding_size}. "
            f"Using first {expected_tokens} tokens for visualization."
        )
        if image_embedding_size < expected_tokens:
            raise ValueError(
                f"Not enough tokens: need at least {expected_tokens}, got {image_embedding_size}"
            )
        # Use only the first expected_tokens for visualization
        actual_image_tokens = expected_tokens
    else:
        actual_image_tokens = image_embedding_size

    # Average attention across heads
    attn = square_attention[0].mean(dim=0).cpu().numpy()  # [seq_len, seq_len]

    # Get generated token range
    num_generated = attn.shape[0] - base_seq_len
    if num_generated <= 0:
        raise ValueError("No generated tokens found")

    # Calculate subplot grid layout
    num_rows = (num_generated + num_per_row - 1) // num_per_row

    # Combined heatmap dimensions
    combined_height = tile_height * grid_size
    combined_width = tile_width * grid_size

    # Get spatial token ranges for all tiles with offset applied
    spatial_ranges = get_spatial_token_ranges(grid_size, tile_width, tile_height, tile_offset)

    # Debug: Log tile token ranges (only once)
    print(f"\n=== Debug: Tile Token Ranges ===")
    print(f"Spatial ranges: {spatial_ranges}")
    print(f"Tile offset: {tile_offset}")
    print(f"Total image tokens used: {actual_image_tokens}")
    print(f"Tokens per tile: {tokens_per_tile}")

    fig, axes = plt.subplots(num_rows, num_per_row, figsize=(4*num_per_row, 4*num_rows), dpi=200)
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    plt.subplots_adjust(wspace=0.3, hspace=0.4)

    # For each generated token
    for i in range(num_generated):
        row_idx = i // num_per_row
        col_idx = i % num_per_row
        ax = axes[row_idx, col_idx]

        token_pos = base_seq_len + i

        # Get attention from this token to all image tokens
        image_attn = attn[token_pos, :]

        # Create combined attention map for all tiles
        combined_attn = np.zeros((combined_height, combined_width))

        # Process each tile (sweep horizontally first)
        for tile_idx in range(tile_width * tile_height):
            # Tile position in grid (row-major order)
            tile_row = tile_idx // tile_width
            tile_col = tile_idx % tile_width

            # Get spatial token range for this tile (offset already applied in get_spatial_token_ranges)
            spatial_start, spatial_end = spatial_ranges[tile_idx]

            # Extract spatial tokens using the pre-calculated range
            spatial_attn = image_attn[spatial_start:spatial_end]

            # Debug: Log max token index for this tile (only for first generated token)
            if i == 0:
                max_attn_idx = spatial_start + spatial_attn.argmax()
                max_attn_val = spatial_attn.max()
                actual_tile_idx = tile_offset + tile_idx
                print(f"  Tile [{tile_row},{tile_col}] (global_idx={actual_tile_idx}): "
                      f"tokens [{spatial_start}:{spatial_end-1}], max_idx={max_attn_idx}, max_val={max_attn_val:.6f}")

            assert len(spatial_attn) == grid_size * grid_size, \
                f"Tile {tile_idx}: expected {grid_size*grid_size} tokens, got {len(spatial_attn)}"

            # Reshape to grid_size × grid_size
            tile_map = spatial_attn.reshape(grid_size, grid_size)

            # Place in combined map
            h_start = tile_row * grid_size
            h_end = h_start + grid_size
            w_start = tile_col * grid_size
            w_end = w_start + grid_size
            combined_attn[h_start:h_end, w_start:w_end] = tile_map

        # Normalize combined attention
        if combined_attn.sum() > 0:
            combined_attn = combined_attn / combined_attn.sum()

        # Decode token text
        if i < len(output_ids):
            token_text = tokenizer.decode(output_ids[i], skip_special_tokens=False).strip()
            if not token_text:
                token_text = f"[{output_ids[i]}]"
        else:
            token_text = f"Token {i}"

        # Visualize combined attention map
        im = ax.imshow(combined_attn, cmap='hot', interpolation='nearest', vmin=0)
        ax.set_title(f"{i}: {token_text}", fontsize=8, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])

        # Draw tile boundaries (cyan lines)
        for t in range(1, tile_height):
            ax.axhline(y=t*grid_size - 0.5, color='cyan', linewidth=1, alpha=0.5)
        for t in range(1, tile_width):
            ax.axvline(x=t*grid_size - 0.5, color='cyan', linewidth=1, alpha=0.5)

        # Add colorbar for first few plots
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    # Hide extra subplots
    for i in range(num_generated, num_rows * num_per_row):
        row_idx = i // num_per_row
        col_idx = i % num_per_row
        axes[row_idx, col_idx].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\n=== Image Attention Per Token Saved ===")
    print(f"Saved to: {output_path}")
    print(f"Tile grid: {tile_height}×{tile_width} tiles")
    print(f"Combined spatial map: {combined_height}×{combined_width}")
    print(f"(Tiles swept horizontally first, cyan lines show boundaries)")
    plt.close()


def visualize_attention_map(attention_matrix, image_end, prompt_end, output_path="attention_map.png", gamma_factor=2.2):
    """
    Visualize the attention matrix as a heatmap with region annotations.

    Args:
        attention_matrix: Tensor of shape [batch, heads, seq_len, seq_len]
        image_end: End index of image embedding region
        prompt_end: End index of prompt text region
        output_path: Path to save the visualization
        gamma_factor: Gamma correction factor for enhancing low attention regions (default: 2.0)
    """
    import numpy as np

    # Take the first batch and average over all heads
    if attention_matrix.dim() == 4:
        # Average over heads: [batch, heads, seq_len, seq_len] -> [seq_len, seq_len]
        attn = attention_matrix[0].mean(dim=0).cpu().numpy()
    else:
        attn = attention_matrix.cpu().numpy()

    # Apply gamma correction to enhance low attention regions
    enhanced_attn = np.power(attn, 1 / gamma_factor)

    seq_len = enhanced_attn.shape[0]

    # Create figure with larger size for better visibility
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot heatmap with gamma-corrected attention
    im = ax.imshow(enhanced_attn, cmap='viridis', aspect='auto', interpolation='nearest')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', rotation=270, labelpad=20)

    # Add grid lines to separate regions
    if image_end > 0:
        ax.axhline(y=image_end - 0.5, color='red', linewidth=2, linestyle='--', alpha=0.7)
        ax.axvline(x=image_end - 0.5, color='red', linewidth=2, linestyle='--', alpha=0.7)

    ax.axhline(y=prompt_end - 0.5, color='yellow', linewidth=2, linestyle='--', alpha=0.7)
    ax.axvline(x=prompt_end - 0.5, color='yellow', linewidth=2, linestyle='--', alpha=0.7)

    # Add labels
    ax.set_xlabel('Key Position', fontsize=12)
    ax.set_ylabel('Query Position', fontsize=12)
    ax.set_title(f'Attention Map (Lower Triangular) - Gamma={gamma_factor:.1f}', fontsize=14, fontweight='bold')

    # Create legend
    legend_elements = []
    if image_end > 0:
        legend_elements.append(mpatches.Patch(color='red', alpha=0.3, label=f'Image Tokens (0-{image_end})'))
    legend_elements.extend([
        mpatches.Patch(color='yellow', alpha=0.3, label=f'Prompt Tokens ({image_end}-{prompt_end})'),
        mpatches.Patch(color='green', alpha=0.3, label=f'Generated Tokens ({prompt_end}-{seq_len})')
    ])
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # Add region annotations
    text_props = dict(fontsize=9, fontweight='bold', ha='center', va='center')

    if image_end > 0:
        # Image region
        ax.text(image_end / 2, image_end / 2, 'Image\nAttention',
                color='white', bbox=dict(boxstyle='round', facecolor='red', alpha=0.5), **text_props)

    # Prompt region
    prompt_center = (image_end + prompt_end) / 2
    ax.text(prompt_center, prompt_center, 'Prompt\nAttention',
            color='white', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5), **text_props)

    # Generated region
    gen_center = (prompt_end + seq_len) / 2
    ax.text(gen_center, gen_center, 'Generation\nAttention',
            color='white', bbox=dict(boxstyle='round', facecolor='green', alpha=0.5), **text_props)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n=== Visualization Saved ===")
    print(f"Attention map saved to: {output_path}")
    plt.close()

def get_schema_from_python_path(path: str) -> str:
    schema_path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location("schema_module", schema_path)
    schema_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(schema_module)

    # Get the Main class from the loaded module
    Main = schema_module.Main
    assert issubclass(
        Main, BaseModel
    ), f"The provided python file {path} does not contain a class Main that describes a JSON schema"
    return Main.schema_json()


def decode_time_token(text: str, *, duration: float, num_time_tokens: int, time_token_format: str) -> str:
    """Replace time tokens in text with actual timestamps."""
    for t in range(num_time_tokens):
        time_token = time_token_format.format(t=t)
        timestamp = round(t * duration / (num_time_tokens - 1), 2)
        text = text.replace(time_token, f"<{timestamp}>")

    # Handle out-of-range time tokens
    excess_pattern = re.compile(rf"<t(\d+)>")
    matches = excess_pattern.findall(text)
    for match in matches:
        t = int(match)
        if t >= num_time_tokens:
            timestamp = round(duration, 2)  # Map to the end of the video
            text = text.replace(f"<t{t}>", f"<{timestamp}>")

    return text


def configure_ps3_and_context_length(model):
    """Configure PS3 settings and adjust context length based on those settings."""

    # get PS3 configs from environment variables
    num_look_close = os.environ.get("NUM_LOOK_CLOSE", None)
    num_token_look_close = os.environ.get("NUM_TOKEN_LOOK_CLOSE", None)
    select_num_each_scale = os.environ.get("SELECT_NUM_EACH_SCALE", None)
    look_close_mode = os.environ.get("LOOK_CLOSE_MODE", None)
    smooth_selection_prob = os.environ.get("SMOOTH_SELECTION_PROB", None)

    # Set PS3 configs
    if num_look_close is not None:
        print("Num look close:", num_look_close)
        num_look_close = int(num_look_close)
        model.num_look_close = num_look_close
    if num_token_look_close is not None:
        print("Num token look close:", num_token_look_close)
        num_token_look_close = int(num_token_look_close)
        model.num_token_look_close = num_token_look_close
    if select_num_each_scale is not None:
        print("Select num each scale:", select_num_each_scale)
        select_num_each_scale = [int(x) for x in select_num_each_scale.split("+")]
        model.get_vision_tower().vision_tower.vision_model.max_select_num_each_scale = select_num_each_scale
    if look_close_mode is not None:
        print("Look close mode:", look_close_mode)
        model.look_close_mode = look_close_mode
    if smooth_selection_prob is not None:
        print("Smooth selection prob:", smooth_selection_prob)
        if smooth_selection_prob.lower() == "true":
            smooth_selection_prob = True
        elif smooth_selection_prob.lower() == "false":
            smooth_selection_prob = False
        else:
            raise ValueError(f"Invalid smooth selection prob: {smooth_selection_prob}")
        model.smooth_selection_prob = smooth_selection_prob

    # Adjust the max context length based on the PS3 config
    context_length = model.tokenizer.model_max_length
    if num_look_close is not None:
        context_length = max(context_length, num_look_close * 2560 // 4 + 1024)
    if num_token_look_close is not None:
        context_length = max(context_length, num_token_look_close // 4 + 1024)
    context_length = max(getattr(model.tokenizer, "model_max_length", context_length), context_length)
    model.config.model_max_length = context_length
    model.config.tokenizer_model_max_length = context_length
    model.llm.config.model_max_length = context_length
    model.llm.config.tokenizer_model_max_length = context_length
    model.tokenizer.model_max_length = context_length


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", "-m", type=str, required=True)
    parser.add_argument("--lora-path", "-l", type=str, default=None)
    parser.add_argument("--conv-mode", "-c", type=str, default="auto")
    parser.add_argument("--text", type=str)
    parser.add_argument("--media", type=str, nargs="+")
    parser.add_argument("--json-mode", action="store_true")
    parser.add_argument("--json-schema", type=str, default=None)
    parser.add_argument("--layer-start", type=int, default=0, help="Starting layer index (default: 0)")
    parser.add_argument("--layer-end", type=int, default=27, help="Ending layer index (default: 27)")
    args = parser.parse_args()

    # Convert json mode to response format
    if not args.json_mode:
        response_format = None
    elif args.json_schema is None:
        response_format = ResponseFormat(type="json_object")
    else:
        schema_str = get_schema_from_python_path(args.json_schema)
        print(schema_str)
        response_format = ResponseFormat(type="json_schema", json_schema=JsonSchemaResponseFormat(schema=schema_str))

    # Load model
    if args.lora_path is None:
        model = llava.load(args.model_path, model_base=None)
    else:
        model = llava.load(args.lora_path, model_base=args.model_path)

    # Configure PS3 and adjust context length
    configure_ps3_and_context_length(model)

    # Set conversation mode
    clib.default_conversation = clib.conv_templates[args.conv_mode].copy()

    # Prepare multi-modal prompt
    prompt = []
    if args.media is not None:
        for media in args.media or []:
            if any(media.endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
                media = Image(media)
            else:
                raise ValueError(f"Unsupported media type: {media}")
            prompt.append(media)
    if args.text is not None:
        prompt.append(args.text)

    # Analyze token positions before generation
    token_info = analyze_token_positions(model, prompt)

    # Configure model for attention capture
    model.llm.config.use_cache=False
    model.llm.config.output_attentions=True

    # Loop through layers
    for layer_idx in range(args.layer_start, args.layer_end + 1):
        print(f"\n{'='*60}")
        print(f"Processing Layer {layer_idx}")
        print(f"{'='*60}")

        # Add hooks for this layer
        hook = add_act_hooks(model, layer_idx)

        # Generate response
        response = model.generate_content(prompt, response_format=response_format)
        print(colored(response, "cyan", attrs=["bold"]))

        # Process attention tensors into square lower triangular matrix
        llm_layer = model.llm.model.layers[layer_idx].self_attn
        acts_list = llm_layer.acts_list

        if len(acts_list) > 0:
            print(f"\n=== Attention Tensor Processing (Layer {layer_idx}) ===")
            print(f"Number of attention tensors captured: {len(acts_list)}")

            first_tensor = acts_list[0]
            batch_size, num_heads, base_seq_len, _ = first_tensor.shape

            # Calculate final sequence length
            final_seq_len = base_seq_len + (len(acts_list) - 1)
            num_generated_tokens = len(acts_list) - 1

            print(f"\nSequence breakdown:")
            print(f"  Base sequence length (image + prompt): {base_seq_len}")
            print(f"  Number of generated tokens: {num_generated_tokens}")
            print(f"  Final sequence length: {final_seq_len}")

            # Estimate image embedding size
            # Image tokens are replaced by image embeddings during forward pass
            num_image_tokens_in_input = len(token_info["image_token_positions"])
            num_text_tokens_in_input = len(token_info["text_token_positions"])

            # The base_seq_len includes expanded image embeddings
            # Estimate: base_seq_len = text_tokens + (image_embeddings per image token)
            if num_image_tokens_in_input > 0:
                image_embedding_size = base_seq_len - num_text_tokens_in_input
            else:
                image_embedding_size = 0

            print(f"\nToken position breakdown:")
            print(f"  Image embedding tokens: 0 to {image_embedding_size-1} ({image_embedding_size} tokens)")
            print(f"  Prompt text tokens: {image_embedding_size} to {base_seq_len-1} ({num_text_tokens_in_input} tokens)")
            print(f"  Generated tokens: {base_seq_len} to {final_seq_len-1} ({num_generated_tokens} tokens)")

            # Create square tensor filled with zeros
            square_attention = torch.zeros(
                batch_size, num_heads, final_seq_len, final_seq_len,
                dtype=first_tensor.dtype, device=first_tensor.device
            )

            # Fill in the base square matrix (top-left)
            square_attention[:, :, :base_seq_len, :base_seq_len] = first_tensor

            # Fill in each subsequent row
            for i, tensor in enumerate(acts_list[1:], start=0):
                row_idx = base_seq_len + i
                _, _, _, width = tensor.shape
                square_attention[:, :, row_idx:row_idx+1, :width] = tensor

            print(f"\nFinal square attention shape: {square_attention.shape}")

            # Tokenize the response to get output token IDs
            response_token_ids = model.tokenizer.encode(response, add_special_tokens=False)

            # Visualize the full attention map
            visualize_attention_map(
                attention_matrix=square_attention,
                image_end=image_embedding_size,
                prompt_end=base_seq_len,
                output_path=f"attn_map/layer_{layer_idx}.png"
            )

            # Visualize image attention for each generated token as 11x11 spatial maps
            visualize_image_attention_per_token(
                square_attention=square_attention,
                image_embedding_size=image_embedding_size,
                base_seq_len=base_seq_len,
                tokenizer=model.tokenizer,
                output_ids=response_token_ids,
                output_path=f"attn_map/image_per_token_layer_{layer_idx}.png",
                tile_width=4,
                tile_height=3,
                tile_offset=0,
            )

            # Visualize single tile (tile 12) with higher detail
            visualize_image_attention_per_token(
                square_attention=square_attention,
                image_embedding_size=image_embedding_size,
                base_seq_len=base_seq_len,
                tokenizer=model.tokenizer,
                output_ids=response_token_ids,
                output_path=f"attn_map/image_per_token_tile12_layer_{layer_idx}.png",
                tile_width=1,
                tile_height=1,
                tile_offset=12,
            )

        # Remove hooks for this layer
        remove_act_hooks(model, layer_idx, hook)
        print(f"\nCompleted processing layer {layer_idx}")


if __name__ == "__main__":
    main()
