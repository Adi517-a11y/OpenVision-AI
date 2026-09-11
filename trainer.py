import os
import json
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

from .model import OpenVisionDenoiser
from .text_encoder import OpenVisionTextEncoder


class OpenVisionDataset(Dataset):

    def __init__(
        self,
        metadata_file,
        image_size=64
    ):
        self.items = []

        with open(
            metadata_file,
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                if not line.strip():
                    continue

                item = json.loads(line)

                if (
                    "image" in item
                    and "caption" in item
                ):
                    self.items.append(item)

        self.transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5]
            )
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):

        item = self.items[index]

        image = Image.open(
            item["image"]
        ).convert("RGB")

        image = self.transform(image)

        return {
            "image": image,
            "caption": item["caption"]
        }


class DiffusionSchedule:

    def __init__(
        self,
        steps=1000,
        device="cpu"
    ):
        self.steps = steps

        beta_start = 0.0001
        beta_end = 0.02

        self.betas = torch.linspace(
            beta_start,
            beta_end,
            steps,
            device=device
        )

        self.alphas = 1.0 - self.betas

        self.alpha_bars = torch.cumprod(
            self.alphas,
            dim=0
        )

    def add_noise(
        self,
        images,
        noise,
        timesteps
    ):

        alpha_bar = self.alpha_bars[
            timesteps
        ]

        alpha_bar = alpha_bar.view(
            -1, 1, 1, 1
        )

        noisy_images = (
            torch.sqrt(alpha_bar) * images
            +
            torch.sqrt(1.0 - alpha_bar) * noise
        )

        return noisy_images


def train(
    metadata_file,
    output_file="openvision_weights.pt",
    epochs=1,
    batch_size=2,
    learning_rate=1e-4,
    image_size=64
):

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    dataset = OpenVisionDataset(
        metadata_file,
        image_size=image_size
    )

    if len(dataset) == 0:
        raise RuntimeError(
            "Dataset contains no training examples."
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    text_encoder = OpenVisionTextEncoder()

    text_encoder = text_encoder.to(device)

    model = OpenVisionDenoiser(
        image_channels=3,
        text_dimension=text_encoder.dimension
    )

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate
    )

    schedule = DiffusionSchedule(
        device=device
    )

    model.train()

    for epoch in range(epochs):

        total_loss = 0.0

        for batch in loader:

            images = batch["image"].to(device)

            captions = batch["caption"]

            with torch.no_grad():

                text_embeddings = text_encoder(
                    captions,
                    device
                )

            text_embeddings = (
                text_embeddings.mean(dim=1)
            )

            noise = torch.randn_like(images)

            timesteps = torch.randint(
                0,
                schedule.steps,
                (images.shape[0],),
                device=device
            )

            noisy_images = schedule.add_noise(
                images,
                noise,
                timesteps
            )

            predicted_noise = model(
                noisy_images,
                timesteps,
                text_embeddings
            )

            loss = F.mse_loss(
                predicted_noise,
                noise
            )

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            total_loss += loss.item()

        average_loss = (
            total_loss / len(loader)
        )

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"Loss: {average_loss:.6f}"
        )

    torch.save(
        model.state_dict(),
        output_file
    )

    print(
        f"Saved OpenVision weights to "
        f"{output_file}"
    )


if __name__ == "__main__":

    train(
        metadata_file="dataset/metadata.jsonl",
        output_file="openvision_weights.pt",
        epochs=1
    )
