import torch
import torch.nn as nn

class DINOv3Extractor(nn.Module):
    def __init__(
        self,
        repo_dir: str,
        model_name: str = "dinov3_vitb16",
        weights: str = None, #[inseire url dei pesi]
        device: str = "cuda",
    ):
        super().__init__()

        self.model = torch.hub.load(
            repo_dir,
            model_name,
            source="local",
            weights=weights,
        ).to(device).eval()

        # ---- metadata ----
        self.patch_size = self.model.patch_size
        self.embed_dim = self.model.embed_dim
        self.device = device

        print(
            f"✅ DINOv3 loaded: {model_name} "
            f"(patch={self.patch_size}, dim={self.embed_dim})"
        )

    @torch.no_grad()
    def forward(self, x):
        """
        x: (B, 3, H, W)
        returns: (B, C, H_patch, W_patch)
        """
        B, _, H, W = x.shape

        tokens = self.model(x)              # (B, 1 + N, C)
        patch_tokens = tokens[:, 1:, :]     # remove CLS

        h_patches = H // self.patch_size
        w_patches = W // self.patch_size

        patch_tokens = (
            patch_tokens
            .transpose(1, 2)
            .reshape(B, self.embed_dim, h_patches, w_patches)
        )

        return patch_tokens
