import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer


class OpenVisionTextEncoder(nn.Module):

    def __init__(
        self,
        model_name="openai/clip-vit-base-patch32"
    ):
        super().__init__()

        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_name
        )

        self.encoder = CLIPTextModel.from_pretrained(
            model_name
        )

        self.dimension = self.encoder.config.hidden_size

    def forward(self, prompts, device):

        tokens = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )

        tokens = {
            key: value.to(device)
            for key, value in tokens.items()
        }

        output = self.encoder(**tokens)

        return output.last_hidden_state
