import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"

import torch
import torch.nn as nn
from transformers import AutoProcessor, SiglipVisionModel

from hyperparameters import n_embd


class SigLIP2VisionEncoder(nn.Module):

    def __init__(
        self,
        model_name="google/siglip2-base-patch16-256",
        local_dir="./models/siglip2",
        freeze=True,
    ):
        super().__init__()
        self.freeze = freeze

        if os.path.isdir(local_dir) and os.listdir(local_dir):
            load_from = local_dir
        else:
            load_from = model_name

        self.processor = AutoProcessor.from_pretrained(load_from)
        self.vision_model = SiglipVisionModel.from_pretrained(load_from)

        if load_from != local_dir:
            os.makedirs(local_dir, exist_ok=True)
            self.processor.save_pretrained(local_dir)
            self.vision_model.save_pretrained(local_dir)

        siglip_dim = self.vision_model.config.hidden_size
        self.projector = nn.Linear(siglip_dim, n_embd)

        if self.freeze:
            for p in self.vision_model.parameters():
                p.requires_grad = False
            self.vision_model.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.vision_model.eval()
        return self

    def preprocess(self, images):
        return self.processor(images=images, return_tensors="pt")

    def forward(self, images):
        if isinstance(images, dict):
            inputs = images
        else:
            inputs = self.processor(images=images, return_tensors="pt")


        param_device = next(self.vision_model.parameters()).device
        inputs = {k: v.to(param_device) for k, v in inputs.items()}

        if self.freeze:
            with torch.no_grad():
                patch_features = self.vision_model(**inputs).last_hidden_state
        else:
            patch_features = self.vision_model(**inputs).last_hidden_state

        projected = self.projector(patch_features)
        return projected


if __name__ == "__main__":
    from PIL import Image

    enc = SigLIP2VisionEncoder()
    enc.eval()

    imgs = [Image.new("RGB", (256, 256), color="gray") for _ in range(2)]

    with torch.no_grad():
        out = enc(imgs)

    raw = enc.vision_model(**{k: v for k, v in enc.preprocess(imgs).items()})
    print("patch features:", tuple(raw.last_hidden_state.shape))
    print("projected:     ", tuple(out.shape))