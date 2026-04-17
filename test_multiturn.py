#!/usr/bin/env python3
"""
Test script for multi-turn conversation support in server.py
"""
import requests
import json
import base64
import os
import re
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

# Server configuration
SERVER_URL = "http://localhost:8001/chat/completions"
MODEL_NAME = "Molmo-7B-O-0924"  # Adjust to match your model


def image_to_base64_url(image_path: str) -> str:
    """
    Convert a local image file to base64 data URL format.

    Args:
        image_path: Path to the local image file

    Returns:
        A data URL in the format: data:image/<type>;base64,<base64_data>
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Determine image type from file extension
    ext = os.path.splitext(image_path)[1].lower()
    if ext in ['.jpg', '.jpeg']:
        image_type = 'jpeg'
    elif ext == '.png':
        image_type = 'png'
    elif ext == '.gif':
        image_type = 'gif'
    elif ext == '.webp':
        image_type = 'webp'
    else:
        # Default to jpeg
        image_type = 'jpeg'

    # Read and encode the image file
    with open(image_path, 'rb') as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Return as data URL
    return f"data:image/{image_type};base64,{image_data}"


def extract_point_from_molmo_response(response_text, image_width, image_height, extract_all=False):
    """
    Extract point(s) from Molmo's response.
    Supports both XML format (<point x="..." y="..."/>) and tuple format (x, y).
    - XML format: coordinates in percent (0-100)
    - Tuple format: coordinates normalized (0-1)

    Args:
        response_text: The response from Molmo
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels
        extract_all: If True, return all points found. If False, return only the first point.

    Returns:
        If extract_all=False: [x, y] coordinates in pixel space (single point)
        If extract_all=True: [[x1, y1], [x2, y2], ...] list of coordinates (multiple points)
    """
    all_points = []

    # Try to find <point> tags in the response (XML format)
    point_pattern = r"<point\b[^>]*>"
    point_matches = re.findall(point_pattern, response_text, flags=re.IGNORECASE)

    if not point_matches:
        # Try to find self-closing tags
        point_pattern = r"<point\b[^>]*/\s*>"
        point_matches = re.findall(point_pattern, response_text, flags=re.IGNORECASE)

    if point_matches:
        # Parse XML format (coordinates in 0-100 range)
        for point_tag in point_matches:
            try:
                # Try to parse as XML
                # Add closing tag if not self-closing
                if not point_tag.endswith("/>"):
                    point_tag_xml = point_tag + "</point>"
                else:
                    point_tag_xml = point_tag
                el = ET.fromstring(point_tag_xml)
            except ET.ParseError:
                # Fallback: use regex to extract x and y attributes
                x_match = re.search(r'x=["\']?([0-9.]+)["\']?', point_tag)
                y_match = re.search(r'y=["\']?([0-9.]+)["\']?', point_tag)

                if not x_match or not y_match:
                    continue  # Skip this point if we can't parse it

                x_percent = float(x_match.group(1))
                y_percent = float(y_match.group(1))
            else:
                # Successfully parsed as XML
                x_str = el.attrib.get("x")
                y_str = el.attrib.get("y")

                if x_str is None or y_str is None:
                    continue  # Skip this point if missing attributes

                x_percent = float(x_str)
                y_percent = float(y_str)

            # Convert from percent (0-100) to pixel coordinates
            x_pixel = int((x_percent / 100.0) * image_width)
            y_pixel = int((y_percent / 100.0) * image_height)
            all_points.append([x_pixel, y_pixel])

    else:
        # Try to parse tuple format: (x, y) - coordinates in 0-1 range
        tuple_pattern = r'\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)'
        tuple_matches = re.findall(tuple_pattern, response_text)

        if not tuple_matches:
            if not extract_all:
                raise ValueError(f"No point found in Molmo response (tried XML and tuple formats): {response_text}")
            else:
                return []  # Return empty list if no points found

        for tuple_match in tuple_matches:
            x_normalized = float(tuple_match[0])
            y_normalized = float(tuple_match[1])

            # Convert from normalized (0-1) to pixel coordinates
            x_pixel = int(x_normalized * image_width)
            y_pixel = int(y_normalized * image_height)
            all_points.append([x_pixel, y_pixel])

    if not all_points:
        if extract_all:
            return []
        else:
            raise ValueError(f"No point found in Molmo response: {response_text}")

    # Return all points or just the first one
    if extract_all:
        return all_points
    else:
        return all_points[0]


def test_single_turn():
    """Test single turn conversation"""
    print("\n=== Testing Single Turn ===")

    request = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": "What is the capital of France?"
            }
        ],
        "max_tokens": 200,
        "temperature": 0.2
    }

    response = requests.post(SERVER_URL, json=request)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {result['choices'][0]['message']['content']}")
    else:
        print(f"Error: {response.text}")


def test_multi_turn():
    """Test multi-turn conversation"""
    print("\n=== Testing Multi-Turn ===")

    request = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": "What is the capital of France?"
            },
            {
                "role": "assistant",
                "content": "The capital of France is Paris."
            },
            {
                "role": "user",
                "content": "What is its population?"
            }
        ],
        "max_tokens": 200,
        "temperature": 0.2
    }

    response = requests.post(SERVER_URL, json=request)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {result['choices'][0]['message']['content']}")
    else:
        print(f"Error: {response.text}")


def test_multi_turn_with_image(task_index=0, eval_index=0):
    """Test multi-turn conversation with an image"""
    print(f"\n=== Testing task_{task_index}/eval_{eval_index} ===")

    # prompt_step1 = "What is the destination point on the target surface where the object should be placed?"
    # prompt_step2 = "point_qa: What is the destination point? Output only one <Point/> tag."

    prompt_step1 = "What is the object to pick-up?"
    prompt_step2 = "point_qa: What is the pick-up point? Output only one <Point/> tag."

    image_path = f"../LIBERO/outputs/render_only/libero_spatial/task_{task_index}/eval_{eval_index}/agentview.png"
    instruction_path = f"../LIBERO/outputs/render_only/libero_spatial/task_{task_index}/eval_{eval_index}/instruction.txt"

    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    # Check if instruction file exists and read it
    if not os.path.exists(instruction_path):
        print(f"Instruction file not found: {instruction_path}")
        return

    with open(instruction_path, 'r') as f:
        prompt = f.read().strip()
    print(f"Loaded instruction: {prompt}")

    # Create output directory
    output_dir = f"outputs/multiturn_results/task_{task_index}/eval_{eval_index}"
    os.makedirs(output_dir, exist_ok=True)

    # Convert image to base64 data URL
    print(f"Converting image to base64: {image_path}")
    image_base64_url = image_to_base64_url(image_path)
    print(f"Base64 data URL created (length: {len(image_base64_url)} chars)")

    # Save the instruction to output directory
    prompt_file = os.path.join(output_dir, "instruction.txt")
    with open(prompt_file, 'w') as f:
        f.write(prompt)
    print(f"Saved instruction to: {prompt_file}")

    # First message with image
    request_1 = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_base64_url
                        }
                    },
                    {
                        "type": "text",
                        "text": f"{prompt}. {prompt_step1}"
                    }
                ]
            }
        ],
        "max_tokens": 200,
        "temperature": 0.2
    }

    print("\n[Turn 1]")
    response_1 = requests.post(SERVER_URL, json=request_1)
    print(f"Status: {response_1.status_code}")

    if response_1.status_code != 200:
        print(f"Error: {response_1.text}")
        return

    result_1 = response_1.json()
    assistant_response_1 = result_1['choices'][0]['message']['content']
    print(f"Assistant: {assistant_response_1}")

    # Second message - multi-turn follow-up
    request_2 = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_base64_url
                        }
                    },
                    {
                        "type": "text",
                        "text": f"{prompt}. {prompt_step1}"
                    }
                ]
            },
            {
                "role": "assistant",
                "content": assistant_response_1
            },
            {
                "role": "user",
                "content": prompt_step2
            }
        ],
        "max_tokens": 200,
        "temperature": 0.2
    }

    print("\n[Turn 2]")
    response_2 = requests.post(SERVER_URL, json=request_2)
    print(f"Status: {response_2.status_code}")

    if response_2.status_code == 200:
        result_2 = response_2.json()
        assistant_response_2 = result_2['choices'][0]['message']['content']
        print(f"Assistant: {assistant_response_2}")

        # Parse the response to extract point coordinates
        print("\n[Parsing Point Coordinates]")
        try:
            # Load image to get dimensions
            img = Image.open(image_path)
            image_width, image_height = img.size
            print(f"Image dimensions: {image_width}x{image_height}")

            # (1) Save original image
            original_output_path = os.path.join(output_dir, "original_image.png")
            img.save(original_output_path)
            print(f"Saved original image to: {original_output_path}")

            # Extract point coordinates from the response
            point_coords = extract_point_from_molmo_response(
                assistant_response_2,
                image_width,
                image_height,
                extract_all=False
            )
            print(f"Parsed point coordinates (pixels): {point_coords}")
            print(f"  x: {point_coords[0]}, y: {point_coords[1]}")

            # (2) Save annotated image with destination point in red
            img_annotated = img.copy()
            draw = ImageDraw.Draw(img_annotated)
            # Draw a red circle at the destination point
            radius = 5
            draw.ellipse(
                [point_coords[0] - radius, point_coords[1] - radius,
                 point_coords[0] + radius, point_coords[1] + radius],
                fill='red',
                outline='red'
            )
            annotated_output_path = os.path.join(output_dir, "annotated_image.png")
            img_annotated.save(annotated_output_path)
            print(f"Saved annotated image to: {annotated_output_path}")

            # Save response to file
            response_file = os.path.join(output_dir, "response.txt")
            with open(response_file, 'w') as f:
                f.write(f"Turn 1 Response:\n{assistant_response_1}\n\n")
                f.write(f"Turn 2 Response:\n{assistant_response_2}\n\n")
                f.write(f"Parsed Coordinates: [{point_coords[0]}, {point_coords[1]}]\n")
            print(f"Saved responses to: {response_file}")

        except ValueError as e:
            print(f"Failed to parse point: {e}")
            # Still save the original image and responses
            img = Image.open(image_path)
            original_output_path = os.path.join(output_dir, "original_image.png")
            img.save(original_output_path)

            error_file = os.path.join(output_dir, "error.txt")
            with open(error_file, 'w') as f:
                f.write(f"Error: {e}\n")
                f.write(f"Turn 1 Response:\n{assistant_response_1}\n\n")
                f.write(f"Turn 2 Response:\n{assistant_response_2}\n")
            print(f"Saved error info to: {error_file}")
        except Exception as e:
            print(f"Error during parsing: {e}")
            error_file = os.path.join(output_dir, "error.txt")
            with open(error_file, 'w') as f:
                f.write(f"Error: {e}\n")
            print(f"Saved error info to: {error_file}")
    else:
        print(f"Error: {response_2.text}")
        error_file = os.path.join(output_dir, "error.txt")
        with open(error_file, 'w') as f:
            f.write(f"HTTP Error: {response_2.status_code}\n")
            f.write(f"Response: {response_2.text}\n")
        print(f"Saved error info to: {error_file}")


if __name__ == "__main__":
    print("Testing Multi-Turn Conversation Support")
    print("=" * 50)

    try:
        # Loop through all task_index (0-9) and eval_index (0-9)
        for task_index in range(10):
            for eval_index in range(10):
                try:
                    test_multi_turn_with_image(task_index, eval_index)
                except requests.exceptions.ConnectionError:
                    print(f"\nError: Could not connect to server for task_{task_index}/eval_{eval_index}")
                    print("Make sure server.py is running.")
                    # Save error to file
                    output_dir = f"outputs/multiturn_results/task_{task_index}/eval_{eval_index}"
                    os.makedirs(output_dir, exist_ok=True)
                    error_file = os.path.join(output_dir, "error.txt")
                    with open(error_file, 'w') as f:
                        f.write("Connection error: Could not connect to server\n")
                except Exception as e:
                    print(f"\nError processing task_{task_index}/eval_{eval_index}: {e}")
                    # Save error to file
                    output_dir = f"outputs/multiturn_results/task_{task_index}/eval_{eval_index}"
                    os.makedirs(output_dir, exist_ok=True)
                    error_file = os.path.join(output_dir, "error.txt")
                    with open(error_file, 'w') as f:
                        f.write(f"Error: {e}\n")

        print("\n" + "=" * 50)
        print("Processing complete!")
        print(f"Results saved to: outputs/multiturn_results/")

    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user")
        print("Partial results saved to: outputs/multiturn_results/")
