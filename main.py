import io
import os
import base64
import torch
from PIL import Image
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="OpenVision AI", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt: str


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@app.get("/")
def root():
    return {
        "name": "OpenVision AI",
        "version": "0.1.0",
        "status": "online",
        "device": DEVICE
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "device": DEVICE
    }


@app.post("/generate")
def generate(request: GenerateRequest):
    prompt = request.prompt.strip()

    if not prompt:
        return {"error": "Prompt cannot be empty"}

    # Temporary development response.
    # The trained OpenVision neural generator will replace this
    # once the model and weights are added.
    image = Image.new(
        "RGB",
        (512, 512),
        (30, 30, 35)
    )

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return {
        "prompt": prompt,
        "image": encoded,
        "model": "OpenVision-0.1.0-dev"
    }
