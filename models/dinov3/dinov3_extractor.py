import os
import torch
import torch.nn as nn


class DINOv3Extractor(nn.Module):
    """
    Clean DINOv3 feature extractor.
    Returns patch-level feature maps (B, C, Hf, Wf).
    """

    def __init__(
        self,
        repo_dir: str,
        model_name: str,
        weights: str,
        device: str = "cuda",
    ):
        super().__init__()

        if not os.path.isfile(weights):
            raise FileNotFoundError(f"Weights not found: {weights}")

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        repo_dir = os.path.abspath(repo_dir)
        weights = os.path.abspath(weights)

        # Load model from local hub
        self.model = torch.hub.load(
            repo_dir,
            model_name,
            source="local",
            weights=weights,
        ).to(self.device)

        self.model.eval()

        # DINOv3 ViT-S/16 defaults
        self.patch_size = 16
        self.embed_dim = 384

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W)
        returns: (B, C, Hf, Wf)
        """
        x = x.to(self.device)
        B, _, H, W = x.shape

        # Forward → patch tokens
        tokens = self.model(x)

        # Expected shape: (B, N, C)
        if tokens.dim() != 3:
            raise RuntimeError(
                f"DINOv3 returned invalid shape {tokens.shape}. "
                "Expected (B, N, C) patch tokens."
            )

        h_patches = H // self.patch_size
        w_patches = W // self.patch_size
        expected_patches = h_patches * w_patches

        if tokens.shape[1] != expected_patches:
            raise RuntimeError(
                f"Patch count mismatch: got {tokens.shape[1]}, "
                f"expected {expected_patches}"
            )

        feats = (
            tokens
            .transpose(1, 2)
            .reshape(B, self.embed_dim, h_patches, w_patches)
        )

        return feats
