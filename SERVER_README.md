# Molmo Server

A FastAPI-based server for running Molmo vision-language models with an OpenAI-compatible API.

## Installation

First, ensure you have the required dependencies:

```bash
pip install fastapi uvicorn transformers torch pillow einops torchvision requests anyio
```

## Starting the Server

### Basic Usage

Start the server with the default Molmo-7B-D model:

```bash
python server.py
```

The server will start on `http://0.0.0.0:8001` by default.

### Custom Configuration

Specify a custom model, host, or port:

```bash
python server.py --model-path allenai/Molmo-7B-D-0924 --host 0.0.0.0 --port 8001
```

### Environment Variables

You can also configure the server using environment variables:

```bash
export MOLMO_HOST=0.0.0.0
export MOLMO_PORT=8001
export MOLMO_MODEL_PATH=allenai/Molmo-7B-D-0924
python server.py
```

## Available Models

The server supports all Molmo models from Hugging Face:

- `allenai/Molmo-7B-D-0924` (default) - Best 7B model for demos
- `allenai/Molmo-7B-O-0924` - Most open 7B model
- `allenai/MolmoE-1B-0924` - Efficient 1B mixture-of-experts model
- `allenai/Molmo-72B-0924` - Best 72B model (requires more memory)

## API Usage

### Using the Client Script

The included client script makes it easy to send requests:

```bash
python client.py --image /path/to/image.jpg --prompt "Describe this image"
```

#### Client Options

```bash
python client.py \
  --image /path/to/image.png \
  --prompt "What do you see in this image?" \
  --model molmo-7b-d-0924 \
  --host localhost \
  --port 8001 \
  --max-tokens 200 \
  --stream
```

### Using the API Directly

The server provides an OpenAI-compatible `/chat/completions` endpoint:

```python
import requests
import base64

# Encode image
with open("image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Send request
response = requests.post(
    "http://localhost:8001/chat/completions",
    json={
        "model": "molmo-7b-d-0924",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Describe this image in detail."
                    }
                ]
            }
        ],
        "max_tokens": 200
    }
)

result = response.json()
print(result["choices"][0]["message"]["content"])
```

### Request Format

**Endpoint:** `POST /chat/completions`

**Request Body:**
```json
{
  "model": "molmo-7b-d-0924",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,<base64_encoded_image>"
          }
        },
        {
          "type": "text",
          "text": "Your prompt here"
        }
      ]
    }
  ],
  "max_tokens": 200,
  "stream": false
}
```

**Response:**
```json
{
  "id": "unique-id",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "molmo-7b-d-0924",
  "index": 0,
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "The model's response text..."
      }
    }
  ]
}
```

## Features

- ✅ OpenAI-compatible API format
- ✅ Support for base64-encoded images
- ✅ Support for image URLs
- ✅ Multiple image support per request
- ✅ Streaming responses (simulated)
- ✅ Automatic model download from Hugging Face
- ✅ bfloat16 optimization for efficiency
- ✅ Request queuing with async lock

## Performance Notes

- The server automatically converts the model to bfloat16 for better performance
- Only one request is processed at a time (controlled by global lock)
- The 7B models require approximately 14GB VRAM
- The 72B model requires approximately 144GB VRAM

## Integration with LIBERO

The server can be used with the LIBERO/vlm_manipulation code. Simply point the API client to:

```python
from vlm_manipulation.curobo_utils import VLMPointExtractor

# Initialize with Molmo API
point_extractor = VLMPointExtractor(
    ckpt_path="allenai/Molmo-7B-D-0924",  # Contains "Molmo" in path
    api_host="localhost",
    api_port=8001
)
```

Note: The VLMPointExtractor will detect "Molmo" in the checkpoint path and use API-based inference.

## Troubleshooting

### Out of Memory Errors

If you encounter OOM errors:
1. Use a smaller model (MolmoE-1B)
2. Reduce the number of images per request
3. Lower the max_tokens parameter

### Connection Errors

- Ensure the server is running: `curl http://localhost:8001/`
- Check firewall settings
- Verify the port is not already in use

### Model Download Issues

- Ensure you have a stable internet connection
- Set the Hugging Face cache directory: `export HF_HOME=/path/to/cache`
- Some models may require authentication: `huggingface-cli login`

## References

- [Molmo GitHub Repository](https://github.com/allenai/molmo)
- [Molmo Model Card](https://huggingface.co/allenai/Molmo-7B-D-0924)
- [Molmo Blog Post](https://molmo.allenai.org/blog)
- [Molmo Paper](https://arxiv.org/pdf/2409.17146)
