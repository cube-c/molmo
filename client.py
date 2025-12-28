#!/usr/bin/env python3
"""
Client script to interact with Molmo server.
Usage: python client.py --image <path> --prompt "<text>"
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import requests


def encode_image_to_base64(image_path: str) -> str:
    """Encode a local image file to base64 data URL."""
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")

    # Determine image format from file extension
    ext = Path(image_path).suffix.lower()
    if ext in [".jpg", ".jpeg"]:
        mime_type = "image/jpeg"
    elif ext == ".png":
        mime_type = "image/png"
    else:
        raise ValueError(f"Unsupported image format: {ext}. Use .jpg, .jpeg, or .png")

    return f"data:{mime_type};base64,{encoded}"


def send_request(
    image_path: str,
    prompt: str,
    model_name: str = "Molmo-7B-D-0924",
    host: str = "localhost",
    port: int = 8001,
    max_tokens: int = 200
) -> dict:
    """Send a chat completion request to the Molmo server."""

    # Encode image
    image_data = encode_image_to_base64(image_path)

    # Prepare request payload
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        "max_tokens": max_tokens
    }

    # Send request
    url = f"http://{host}:{port}/chat/completions"

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}", file=sys.stderr)
        if hasattr(e, "response") and e.response is not None:
            print(f"Response: {e.response.text}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Molmo Server Client")
    parser.add_argument("--image", required=True, help="Path to the image file")
    parser.add_argument("--prompt", required=True, help="Text prompt for the model")
    parser.add_argument("--model", default="Molmo-7B-D-0924", help="Model name (default: Molmo-7B-D-0924)")
    parser.add_argument("--host", default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8001, help="Server port (default: 8001)")
    parser.add_argument("--max-tokens", type=int, default=1000, help="Maximum tokens to generate (default: 1000)")

    args = parser.parse_args()

    # Validate image path
    if not Path(args.image).exists():
        print(f"Error: Image file not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    print(f"Sending request to server at {args.host}:{args.port}")
    print(f"Image: {args.image}")
    print(f"Prompt: {args.prompt}")
    print("-" * 50)

    # Send request
    result = send_request(
        image_path=args.image,
        prompt=args.prompt,
        model_name=args.model,
        host=args.host,
        port=args.port,
        max_tokens=args.max_tokens
    )

    # Print response
    print("\nResponse:")
    if "choices" in result and len(result["choices"]) > 0:
        message = result["choices"][0]["message"]
        print(f"{message['role'].capitalize()}: {message['content']}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
