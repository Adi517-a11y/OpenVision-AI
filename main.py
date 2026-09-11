import io
import os
import base64

import torch
from PIL import Image
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openvision.sampler import OpenVisionSampler


app = FastAPI(
    title="OpenVision AI",
    version="0.1.0"
)

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

sampler = None


def load_model():

    global sampler

    weights_path = "openvision_weights.pt"

    if not os.path.exists(weights_path):

        print(
            "OpenVision weights not found."
        )

        return

    try:

        sampler = OpenVisionSampler(
            weights_path
        )

        print(
            "OpenVision generation model ready."
        )

    except Exception as error:

        print(
            f"Model loading failed: {error}"
        )


load_model()


@app.get("/")
def root():

    return {
        "name": "OpenVision AI",
        "version": "0.1.0",
        "status": "online",
        "device": DEVICE,
        "model_loaded": sampler is not None
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "device": DEVICE,
        "model_loaded": sampler is not None
    }


@app.post("/generate")
def generate(request: GenerateRequest):

    prompt = request.prompt.strip()

    if not prompt:

        return {
            "error": "Prompt cannot be empty"
        }

    if sampler is None:

        return {
            "error": (
                "OpenVision model is not trained yet."
            )
        }

    try:

        generated = sampler.generate(
            prompt,
            steps=100
        )

        # --------------------------------------------
        # Convert tensor to PIL image
        # --------------------------------------------

        generated = generated[0]

        generated = (
            generated
            .detach()
            .cpu()
            .permute(1, 2, 0)
            .numpy()
        )

        generated = (
            generated * 255
        ).clip(
            0,
            255
        ).astype(
            "uint8"
        )

        image = Image.fromarray(
            generated
        )

        # --------------------------------------------
        # Encode as PNG
        # --------------------------------------------

        buffer = io.BytesIO()

        image.save(
            buffer,
            format="PNG"
        )

        encoded = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        return {

            "prompt": prompt,

            "image": encoded,

            "model": "OpenVision-0.1.0",

            "resolution": "64x64"
        }

    except Exception as error:

        return {
            "error": str(error)
        }
