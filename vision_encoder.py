import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"

import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor

from hyperparameters import device, n_embd


class SigLIP2VisionEncoder(nn.Module):
    def __init__(self, model_name="./models/siglip2", freeze=True):
        super().__init__()

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.vision_model = AutoModel.from_pretrained(model_name)

        self.processor.save_pretrained("./models/siglip2")  
        self.vision_model.save_pretrained("./models/siglip2")

        siglip_dim = self.vision_model.config.vision_config.hidden_size

        # Converts SigLIP2 image embedding size to your GPT embedding size
        self.projector = nn.Linear(siglip_dim, n_embd)

        if freeze:
            for p in self.vision_model.parameters():
                p.requires_grad = False

    def forward(self, images):
        """
        images: list of PIL images
        returns: projected image token, shape (B, 1, n_embd)
        """

        inputs = self.processor(
            images=images,
            return_tensors="pt"
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        if not any(p.requires_grad for p in self.vision_model.parameters()):
            with torch.no_grad():
                image_features = self.vision_model.get_image_features(**inputs)
        else:
            image_features = self.vision_model.get_image_features(**inputs)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        projected = self.projector(image_features)

        # (B, n_embd) -> (B, 1, n_embd)
        projected = projected.unsqueeze(1)

        return projected