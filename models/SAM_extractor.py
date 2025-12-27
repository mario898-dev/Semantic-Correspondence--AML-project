import torch
import torch.nn as nn
import sys
import os


class SAMExtractor(nn.Module):
    def __init__(self, repo_dir: str, model_type: str = "vit_b", device: str = "cuda"):
        super().__init__()
        self.device = device

        # 1. Setup path ASSOLUTO
        abs_repo_dir = os.path.abspath(repo_dir)
        if abs_repo_dir not in sys.path:
            sys.path.insert(0, abs_repo_dir)

        try:
            from segment_anything import sam_model_registry
        except ImportError as e:
            raise ImportError(
                f"Controlla che {abs_repo_dir} contenga 'segment_anything'. Errore: {e}"
            )

        # 2. Path ai checkpoint
        ckpt = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "checkpoints",
            "SAM",
            f"sam_{model_type}.pth"
        )

        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Pesi SAM non trovati: {ckpt}")

        # 3. Inizializzazione SAM
        sam = sam_model_registry[model_type](checkpoint=ckpt)
        self.model = sam.to(device).eval()

        # Metadata (per compatibilità pipeline)
        #self.patch_size = self.model.image_encoder.patch_embed.proj.kernel_size[0]
        #self.embed_dim = self.model.image_encoder.embed_dim

        print(
            f"SAM caricato: {model_type} "
            f"(patch_size={self.patch_size}, dim={self.embed_dim})"
        )

    @torch.no_grad()
    def forward(self, x):
        # Estrai SOLO le feature dell'image encoder
        feats = self.model.image_encoder(x)
        return feats
