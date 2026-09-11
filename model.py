import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension

    def forward(self, timestep):
        half = self.dimension // 2

        frequencies = torch.exp(
            -math.log(10000)
            * torch.arange(
                half,
                device=timestep.device,
                dtype=torch.float32
            )
            / max(half - 1, 1)
        )

        values = timestep.float().unsqueeze(1) * frequencies.unsqueeze(0)

        return torch.cat(
            [torch.sin(values), torch.cos(values)],
            dim=1
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels, time_dimension):
        super().__init__()

        self.norm1 = nn.GroupNorm(8, channels)

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.time_projection = nn.Linear(
            time_dimension,
            channels
        )

        self.norm2 = nn.GroupNorm(8, channels)

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, x, time_embedding):
        residual = x

        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        time = self.time_projection(time_embedding)
        time = time.unsqueeze(-1).unsqueeze(-1)

        h = h + time

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        return h + residual


class OpenVisionDenoiser(nn.Module):

    def __init__(
        self,
        image_channels=4,
        base_channels=128,
        text_dimension=768,
        time_dimension=256
    ):
        super().__init__()

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dimension),
            nn.Linear(time_dimension, time_dimension),
            nn.SiLU(),
            nn.Linear(time_dimension, time_dimension)
        )

        self.text_projection = nn.Linear(
            text_dimension,
            base_channels
        )

        self.input = nn.Conv2d(
            image_channels,
            base_channels,
            kernel_size=3,
            padding=1
        )

        self.block1 = ResidualBlock(
            base_channels,
            time_dimension
        )

        self.block2 = ResidualBlock(
            base_channels,
            time_dimension
        )

        self.block3 = ResidualBlock(
            base_channels,
            time_dimension
        )

        self.output_norm = nn.GroupNorm(
            8,
            base_channels
        )

        self.output = nn.Conv2d(
            base_channels,
            image_channels,
            kernel_size=3,
            padding=1
        )

    def forward(
        self,
        noisy_image,
        timestep,
        text_embedding
    ):
        time_embedding = self.time_embedding(timestep)

        h = self.input(noisy_image)

        text = self.text_projection(text_embedding)
        text = text.unsqueeze(-1).unsqueeze(-1)

        h = h + text

        h = self.block1(h, time_embedding)
        h = self.block2(h, time_embedding)
        h = self.block3(h, time_embedding)

        h = self.output_norm(h)
        h = F.silu(h)

        return self.output(h)
