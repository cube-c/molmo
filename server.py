import argparse
import base64
import json
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from io import BytesIO
from typing import List, Literal, Optional, Union, get_args

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image as PILImage
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

import asyncio
from anyio.lowlevel import RunVar
from anyio import CapacityLimiter


class TextContent(BaseModel):
    type: Literal["text"]
    text: str


class MediaURL(BaseModel):
    url: str


class ImageContent(BaseModel):
    type: Literal["image_url"]
    image_url: MediaURL


IMAGE_CONTENT_BASE64_REGEX = re.compile(r"^data:image/(png|jpe?g);base64,(.*)$")


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, List[Union[TextContent, ImageContent]]]


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 200
    temperature: Optional[float] = 0.2


model = None
model_name = None
processor = None


def get_timestamp():
    return int(time.time())


def load_image(image_url: str) -> PILImage:
    """Load image from URL or base64 data."""
    if image_url.startswith("http") or image_url.startswith("https"):
        import requests
        response = requests.get(image_url)
        image = PILImage.open(BytesIO(response.content)).convert("RGB")
    else:
        match_results = IMAGE_CONTENT_BASE64_REGEX.match(image_url)
        if match_results is None:
            raise ValueError(f"Invalid image url: {image_url[:64]}")
        image_base64 = match_results.groups()[1]
        image = PILImage.open(BytesIO(base64.b64decode(image_base64))).convert("RGB")
    return image


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_name, processor

    model_path = app.args.model_path
    model_name = model_path.split("/")[-1] if "/" in model_path else model_path

    print(f"Loading Molmo model from {model_path}...")

    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )

    # Convert to bfloat16 for efficiency if CUDA is available
    if torch.cuda.is_available():
        model = model.to(dtype=torch.bfloat16)

    print(f"{model_name=} {model_path=} loaded successfully.")
    print("Setting capacity limiter to 1")
    RunVar("_default_thread_limiter").set(CapacityLimiter(1))

    global globallock
    globallock = asyncio.Lock()

    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def read_root():
    return {"message": "Welcome to the Molmo API. This is for internal use only. Please use /chat/completions for chat completions."}


@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    current_time = time.strftime("%H:%M:%S-%s", time.localtime())
    current_time_hash = uuid.uuid5(uuid.NAMESPACE_DNS, current_time)
    obj_hash = uuid.uuid5(uuid.NAMESPACE_DNS, str(request.dict()))
    print("[Req recv]", current_time_hash, current_time, request.dict().keys())

    try:
        global model, processor

        if request.model != model_name:
            raise ValueError(
                f"The endpoint is configured to use the model {model_name}, "
                f"but the request model is {request.model}"
            )

        # Parse messages to extract images and text
        images = []
        text_parts = []

        for message in request.messages:
            if isinstance(message.content, str):
                text_parts.append(message.content)
            elif isinstance(message.content, list):
                for content in message.content:
                    if content.type == "text":
                        text_parts.append(content.text)
                    elif content.type == "image_url":
                        image = load_image(content.image_url.url)
                        images.append(image)
                    else:
                        raise NotImplementedError(f"Unsupported content type: {content.type}")

        # Combine text parts
        prompt = " ".join(text_parts)

        try:
            # Process inputs
            inputs = processor.process(
                images=images if images else None,
                text=prompt
            )
        except Exception as e:
            raise ValueError(f"Error at line: processor.process() - {str(e)}")

        try:
            # Move inputs to device and create batch
            processed_inputs = {}
            for k, v in inputs.items():
                if v is not None:
                    v = v.to(model.device)
                    # Only unsqueeze if it doesn't already have batch dimension
                    if v.ndim > 0 and v.shape[0] != 1:
                        v = v.unsqueeze(0)
                    processed_inputs[k] = v
            inputs = processed_inputs
        except Exception as e:
            raise ValueError(f"Error at line: inputs processing/unsqueeze - {str(e)}")

        try:
            # Ensure images are in bfloat16 if model is
            if "images" in inputs and inputs["images"] is not None and model.dtype == torch.bfloat16:
                inputs["images"] = inputs["images"].to(torch.bfloat16)
        except Exception as e:
            raise ValueError(f"Error at line: images bfloat16 conversion - {str(e)}")

        try:
            # Store original input length for later
            input_ids = inputs.get('input_ids')
            if input_ids is None:
                raise ValueError("Processor did not return input_ids")
            original_input_length = input_ids.size(-1)
        except Exception as e:
            raise ValueError(f"Error at line: input_ids length extraction - {str(e)}")

        await globallock.acquire()
        try:
            with torch.inference_mode():
                try:
                    # Validate inputs before generation
                    for k, v in inputs.items():
                        if v is None:
                            raise ValueError(f"Input '{k}' is None")

                    # Generate response
                    output = model.generate_from_batch(
                        inputs,
                        GenerationConfig(max_new_tokens=request.max_tokens, stop_strings="<|endoftext|>"),
                        tokenizer=processor.tokenizer
                    )

                    if output is None:
                        raise ValueError("model.generate_from_batch() returned None")
                except Exception as e:
                    raise ValueError(f"Error at line: model.generate_from_batch() - {str(e)}")

                try:
                    # Decode generated tokens
                    generated_tokens = output[0, original_input_length:]
                    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                except Exception as e:
                    raise ValueError(f"Error at line: token decoding - {str(e)}")

            if globallock.locked():
                globallock.release()

            print("\nAssistant: ", generated_text)

            return {
                "id": uuid.uuid4().hex,
                "object": "chat.completion",
                "created": get_timestamp(),
                "model": request.model,
                "index": 0,
                "choices": [
                    {"message": ChatMessage(role="assistant", content=generated_text)}
                ],
            }
        except Exception as e:
            if globallock.locked():
                globallock.release()
            raise

    except Exception as e:
        if globallock.locked():
            globallock.release()

        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


if __name__ == "__main__":
    host = os.getenv("MOLMO_HOST", "0.0.0.0")
    port = os.getenv("MOLMO_PORT", 8001)
    model_path = os.getenv("MOLMO_MODEL_PATH", "allenai/Molmo-7B-D-0924")

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=host)
    parser.add_argument("--port", type=int, default=port)
    parser.add_argument("--model-path", type=str, default=model_path)

    app.args = parser.parse_args()
    port = int(app.args.port)

    uvicorn.run(
        app,
        host=app.args.host,
        port=app.args.port,
        workers=1,
        timeout_keep_alive=60,
    )
