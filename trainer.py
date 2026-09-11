import os
import json
import math

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from .model import OpenVisionDenoiser
from .text_encoder import OpenVisionTextEncoder


# ============================================================
# OpenVision AI — Training Engine
# Version 0.1.0
# ============================================================

IMAGE_SIZE = 64
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
TIMESTEPS = 1000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Dataset
# ============================================================

class OpenVisionDataset(Dataset):

    def __init__(self, metadata_file):

        self.items = []

        with open(
            metadata_file,
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                try:
                    item = json.loads(line)

                    if (
                        "image" in item
                        and "caption" in item
                    ):
                        self.items.append(item)

                except json.JSONDecodeError:
                    continue

        self.transform = transforms.Compose([
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5]
            )
        ])

        print(
            f"Loaded {len(self.items)} training examples."
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):

        item = self.items[index]

        image_path = os.path.join(
            "dataset",
            item["image"]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        image = self.transform(image)

        caption = item["caption"]

        return image, caption


# ============================================================
# Noise Schedule
# ============================================================

def create_noise_schedule():

    betas = torch.linspace(
        0.0001,
        0.02,
        TIMESTEPS,
        device=DEVICE
    )

    alphas = 1.0 - betas

    alpha_bars = torch.cumprod(
        alphas,
        dim=0
    )

    return betas, alphas, alpha_bars


# ============================================================
# Add Noise
# ============================================================

def add_noise(
    images,
    timesteps,
    alpha_bars
):

    noise = torch.randn_like(images)

    selected_alpha_bars = (
        alpha_bars[timesteps]
        .view(-1, 1, 1, 1)
    )

    noisy_images = (
        torch.sqrt(selected_alpha_bars)
        * images
        +
        torch.sqrt(
            1.0 - selected_alpha_bars
        )
        * noise
    )

    return noisy_images, noise


# ============================================================
# Training
# ============================================================

def train(
    metadata_file="dataset/metadata.jsonl",
    output_file="openvision_weights.pt",
    epochs=1
):

    print()
    print("==========================================")
    print("       OPENVISION AI TRAINING")
    print("==========================================")
    print()

    print(f"Device: {DEVICE}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Timesteps: {TIMESTEPS}")
    print(f"Batch size: {BATCH_SIZE}")
    print()

    if not os.path.exists(metadata_file):

        raise FileNotFoundError(
            f"Dataset metadata not found: "
            f"{metadata_file}"
        )

    dataset = OpenVisionDataset(
        metadata_file
    )

    if len(dataset) == 0:

        raise RuntimeError(
            "Dataset is empty. "
            "Run crawler.py first."
        )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    # --------------------------------------------------------
    # Text encoder
    # --------------------------------------------------------

    print("Loading text encoder...")

    text_encoder = OpenVisionTextEncoder()

    text_encoder = text_encoder.to(
        DEVICE
    )

    # Freeze the text encoder for now.
    #
    # This lets us first train the image-generation
    # network against stable language embeddings.

    text_encoder.eval()

    for parameter in text_encoder.parameters():
        parameter.requires_grad = False

    text_dimension = (
        text_encoder.dimension
    )

    # --------------------------------------------------------
    # Image denoiser
    # --------------------------------------------------------

    print("Creating OpenVision denoiser...")

    model = OpenVisionDenoiser(
        image_channels=3,
        base_channels=128,
        text_dimension=text_dimension,
        time_dimension=256
    )

    model = model.to(
        DEVICE
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )

    # --------------------------------------------------------
    # Noise schedule
    # --------------------------------------------------------

    betas, alphas, alpha_bars = (
        create_noise_schedule()
    )

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    total_steps = 0

    for epoch in range(epochs):

        model.train()

        epoch_loss = 0.0

        print()
        print(
            f"===== Epoch {epoch + 1}/{epochs} ====="
        )

        for batch_index, batch in enumerate(
            dataloader
        ):

            images, captions = batch

            images = images.to(
                DEVICE
            )

            # --------------------------------------------
            # Encode captions
            # --------------------------------------------

            with torch.no_grad():

                text_embeddings = (
                    text_encoder(
                        list(captions),
                        DEVICE
                    )
                )

                # CLIP returns one embedding
                # for every token.
                #
                # Average the token representations
                # to create one conditioning vector.

                text_embeddings = (
                    text_embeddings.mean(
                        dim=1
                    )
                )

            # --------------------------------------------
            # Select random diffusion timestep
            # --------------------------------------------

            timesteps = torch.randint(
                0,
                TIMESTEPS,
                (
                    images.shape[0],
                ),
                device=DEVICE
            )

            # --------------------------------------------
            # Add noise
            # --------------------------------------------

            noisy_images, noise = add_noise(
                images,
                timesteps,
                alpha_bars
            )

            # --------------------------------------------
            # Predict noise
            # --------------------------------------------

            predicted_noise = model(
                noisy_images,
                timesteps,
                text_embeddings
            )

            # --------------------------------------------
            # Diffusion loss
            # --------------------------------------------

            loss = F.mse_loss(
                predicted_noise,
                noise
            )

            # --------------------------------------------
            # Backpropagation
            # --------------------------------------------

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            # Prevent extremely large gradients.

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            # --------------------------------------------
            # Statistics
            # --------------------------------------------

            loss_value = loss.item()

            epoch_loss += loss_value

            total_steps += 1

            if batch_index % 10 == 0:

                print(
                    f"Step {total_steps} | "
                    f"Loss: {loss_value:.6f}"
                )

        average_loss = (
            epoch_loss /
            max(len(dataloader), 1)
        )

        print()
        print(
            f"Epoch {epoch + 1} complete."
        )

        print(
            f"Average loss: "
            f"{average_loss:.6f}"
        )

        # ------------------------------------------------
        # Save checkpoint after every epoch
        # ------------------------------------------------

        checkpoint = {

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "epoch":
                epoch + 1,

            "steps":
                total_steps,

            "image_size":
                IMAGE_SIZE,

            "timesteps":
                TIMESTEPS,

            "text_dimension":
                text_dimension,

            "base_channels":
                128
        }

        torch.save(
            checkpoint,
            output_file
        )

        print(
            f"Checkpoint saved: {output_file}"
        )

    print()
    print("==========================================")
    print("       TRAINING FINISHED")
    print("==========================================")
    print()
    print(
        f"Model saved to: {output_file}"
    )
    print(
        f"Total training steps: {total_steps}"
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    train(
        metadata_file="dataset/metadata.jsonl",
        output_file="openvision_weights.pt",
        epochs=1
    )
