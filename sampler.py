import torch
import torch.nn.functional as F

from .model import OpenVisionDenoiser
from .text_encoder import OpenVisionTextEncoder


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 64
TIMESTEPS = 1000


class OpenVisionSampler:

    def __init__(
        self,
        weights_path="openvision_weights.pt"
    ):

        print("Loading OpenVision model...")

        self.text_encoder = OpenVisionTextEncoder()
        self.text_encoder = self.text_encoder.to(DEVICE)
        self.text_encoder.eval()

        self.model = OpenVisionDenoiser(
            image_channels=3,
            base_channels=128,
            text_dimension=self.text_encoder.dimension,
            time_dimension=256
        )

        checkpoint = torch.load(
            weights_path,
            map_location=DEVICE
        )

        # Support both a complete checkpoint and
        # a raw state dictionary.

        if "model_state_dict" in checkpoint:

            self.model.load_state_dict(
                checkpoint["model_state_dict"]
            )

        else:

            self.model.load_state_dict(
                checkpoint
            )

        self.model = self.model.to(DEVICE)
        self.model.eval()

        self.betas = torch.linspace(
            0.0001,
            0.02,
            TIMESTEPS,
            device=DEVICE
        )

        self.alphas = 1.0 - self.betas

        self.alpha_bars = torch.cumprod(
            self.alphas,
            dim=0
        )

        print("OpenVision model loaded.")

    @torch.no_grad()
    def generate(
        self,
        prompt,
        steps=100
    ):

        if not prompt.strip():

            raise ValueError(
                "Prompt cannot be empty."
            )

        print(
            f"Generating: {prompt}"
        )

        # ------------------------------------------------
        # Encode prompt
        # ------------------------------------------------

        text_embedding = self.text_encoder(
            [prompt],
            DEVICE
        )

        text_embedding = (
            text_embedding.mean(dim=1)
        )

        # ------------------------------------------------
        # Start from random noise
        # ------------------------------------------------

        image = torch.randn(
            1,
            3,
            IMAGE_SIZE,
            IMAGE_SIZE,
            device=DEVICE
        )

        # ------------------------------------------------
        # Diffusion sampling
        # ------------------------------------------------

        timestep_values = torch.linspace(
            TIMESTEPS - 1,
            0,
            steps,
            device=DEVICE
        ).long()

        for index, timestep in enumerate(
            timestep_values
        ):

            timestep_batch = timestep.view(
                1
            )

            predicted_noise = self.model(
                image,
                timestep_batch,
                text_embedding
            )

            beta = self.betas[
                timestep
            ]

            alpha = self.alphas[
                timestep
            ]

            alpha_bar = self.alpha_bars[
                timestep
            ]

            # Reverse diffusion update

            image = (
                1 / torch.sqrt(alpha)
            ) * (
                image
                -
                (
                    beta
                    /
                    torch.sqrt(
                        1 - alpha_bar
                    )
                )
                * predicted_noise
            )

            # Add controlled randomness except
            # at the final step.

            if index < len(
                timestep_values
            ) - 1:

                noise = torch.randn_like(
                    image
                )

                image = (
                    image
                    +
                    torch.sqrt(beta)
                    * noise
                )

            if index % 10 == 0:

                print(
                    f"Sampling step "
                    f"{index + 1}/{steps}"
                )

        # ------------------------------------------------
        # Convert model output to image range
        # ------------------------------------------------

        image = image.clamp(
            -1,
            1
        )

        image = (
            (image + 1) / 2
        )

        return image
