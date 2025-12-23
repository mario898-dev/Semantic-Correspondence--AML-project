import torch
import torch.nn as nn
from segment_anything import sam_model_registry

class SAMExtractor(nn.Module):
    def __init__(self, model_type: str, device: str = "cuda"):
        super().__init__()
        self.device = device

        sam = sam_model_registry[model_type](
            checkpoint=f"sam_{model_type}.pth"
        )
        self.image_encoder = sam.image_encoder.to(device).eval()

        # metadata
        self.patch_size = 16          # SAM ViT uses 16×16 patches
        self.embed_dim = 256          # fixed by SAM

        print(
            f"SAM {model_type} loaded "
            f"(patch_size={self.patch_size}, dim={self.embed_dim})"
        )

    @torch.no_grad()
    def forward(self, x):
        """
        x: (B, 3, H, W)  -- già preprocessato dal dataset
        return: (B, 256, H/16, W/16)
        """
        x = x.to(self.device)

        patch_features = self.image_encoder(x)

        return patch_features
