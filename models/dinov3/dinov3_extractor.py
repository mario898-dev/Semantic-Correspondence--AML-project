import os
import torch
import torch.nn as nn

class DINOv3Extractor(nn.Module):
    def __init__(
        self,
        repo_dir: str,
        model_name: str,
        weights: str,
        device: str = "cuda",
    ):
        super().__init__()

        if weights is None or not os.path.isfile(weights):
            raise FileNotFoundError(f"Weights not found: {weights}")

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # IMPORTANT: torch.hub needs an absolute path often works better on Windows
        repo_dir = os.path.abspath(repo_dir)

        self.model = torch.hub.load(
            repo_dir,
            model_name,
            source="local",
            weights=weights,   # DINOv3 hubconf.py expects this
        ).to(self.device).eval()

        self.patch_size = getattr(self.model, "patch_size", 16)
        self.embed_dim = getattr(self.model, "embed_dim", None)

        if self.embed_dim is None:
            # fallback: infer from a dummy forward if needed
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224, device=self.device)
                toks = self.model(dummy)  # (1, 1+N, C)
                self.embed_dim = toks.shape[-1]

        print(f"✅ DINOv3 loaded: {model_name} (patch={self.patch_size}, dim={self.embed_dim})")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W)
        returns: (B, C, H_patch, W_patch)
        """
        x = x.to(self.device)
        B, _, H, W = x.shape

        tokens = self.model(x)          # (B, 1+N, C)
        patch_tokens = tokens[:, 1:, :] # remove CLS

        h_patches = H // self.patch_size
        w_patches = W // self.patch_size

        patch_tokens = (
            patch_tokens
            .transpose(1, 2)
            .reshape(B, self.embed_dim, h_patches, w_patches)
        )
        return patch_tokens
