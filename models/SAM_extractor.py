import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class SAMExtractor(nn.Module):
    """
    SAM feature extractor compatibile con pipeline AML:
    - Input:  (B, 3, H, W)  (tipicamente già ImageNet-normalized da SD4Match)
    - Output: (B, C, Hf, Wf)

    Fix principali:
    1) De-normalizza da ImageNet -> [0,1] e poi applica la normalizzazione SAM (pixel_mean/std in 0..255)
    2) Interpola la pos_embed dell'image encoder alla griglia di patch dell'input (es. 32x32 se IMG_SIZE=512)
       evitando l'errore "32 vs 64" tipico di SAM pretrainato per 1024.
    """

    def __init__(self, repo_dir: str, model_type: str = "vit_b", device: str = "cuda"):
        super().__init__()
        self.device = device

        # Path robusto verso il submodule segment-anything
        abs_repo_dir = os.path.abspath(repo_dir)
        if abs_repo_dir not in sys.path:
            sys.path.insert(0, abs_repo_dir)

        try:
            from segment_anything import sam_model_registry
        except ImportError as e:
            raise ImportError(
                f"Controlla che {abs_repo_dir} contenga 'segment_anything'. Errore: {e}"
            )

        # Checkpoint
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ckpt = os.path.join(repo_root, "checkpoints", "SAM", f"sam_{model_type}.pth")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Pesi SAM non trovati: {ckpt}")

        # Init SAM
        sam = sam_model_registry[model_type](checkpoint=ckpt)
        self.model = sam.to(device).eval()

        # Costanti per de-normalizzazione ImageNet (se l'input arriva già così)
        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        print(f"✅ SAM loaded: {model_type} (checkpoint: {os.path.basename(ckpt)})")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,3,H,W) dal dataset (spesso ImageNet-normalized)
        ritorna: (B,C,Hf,Wf)
        """
        enc = self.model.image_encoder

        # ---------------------------------------------------------
        # 1) Porta l'input in un formato sensato per SAM
        #    SD4Match spesso dà input ImageNet-normalized:
        #    de-normalizzo -> [0,1] -> *255 -> normalizzazione SAM
        # ---------------------------------------------------------
        # euristica: se max > 2, è quasi certamente z-score (ImageNet norm)
        if x.max() > 2.0 or x.min() < -1.0:
            x01 = (x * self.imagenet_std + self.imagenet_mean).clamp(0.0, 1.0)
        else:
            # se già in [0,1] (o simile), clamp e basta
            x01 = x.clamp(0.0, 1.0)

        x255 = x01 * 255.0
        x_sam = (x255 - self.model.pixel_mean) / self.model.pixel_std

        # ---------------------------------------------------------
        # 2) Fix pos_embed: SAM ViT ha pos_embed pretrainato per 1024
        #    Se l'input è 512, griglia patch = 32 e va interpolata.
        # ---------------------------------------------------------
        # Patch stride (per vit_b è 16)
        patch = enc.patch_embed.proj.stride[0]
        hp, wp = x_sam.shape[-2] // patch, x_sam.shape[-1] // patch

        pos = enc.pos_embed  # tipicamente (1, 64, 64, C)
        if pos.shape[1] != hp or pos.shape[2] != wp:
            # (1, H0, W0, C) -> (1, C, H0, W0)
            pos_ = pos.permute(0, 3, 1, 2)
            pos_ = F.interpolate(pos_, size=(hp, wp), mode="bilinear", align_corners=False)
            # (1, C, hp, wp) -> (1, hp, wp, C)
            pos_ = pos_.permute(0, 2, 3, 1)
            # aggiorno pos_embed in modo "frozen" (no grad)
            enc.pos_embed = nn.Parameter(pos_, requires_grad=False)

        # ---------------------------------------------------------
        # 3) Forward encoder -> feature map
        # ---------------------------------------------------------
        feats = enc(x_sam)  # (B, C, hp, wp)
        return feats
