import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class SAMExtractor(nn.Module):
    """
    SAM feature extractor compatibile con pipeline AML:
    - Input:  (B, 3, H, W)  (spesso ImageNet-normalized da SD4Match)
    - Output: (B, C, Hf, Wf)

    Fix principali:
    1) De-normalizza da ImageNet -> [0,1] e poi applica la normalizzazione SAM (pixel_mean/std in 0..255)
    2) Interpola la pos_embed dell'image encoder alla griglia di patch dell'input (es. 32x32 se IMG_SIZE=512)
       evitando l'errore "32 vs 64" tipico di SAM pretrainato per 1024.
    """

    def __init__(self, repo_dir: str, model_type: str = "vit_b", device: str = "cuda"):
        super().__init__()
        self.device = device

        # Path verso il submodule segment-anything
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
        self.model = sam.to(device)

        # Costanti ImageNet (buffer)
        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        print(f"✅ SAM loaded: {model_type} (checkpoint: {os.path.basename(ckpt)})")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc = self.model.image_encoder

        # 1) Input preprocessing per SAM
        # euristica: se max > 2 o min < -1 -> probabile ImageNet z-score
        if x.max() > 2.0 or x.min() < -1.0:
            # FIX DEVICE: porta mean/std sul device di x
            mean = self.imagenet_mean.to(x.device)
            std = self.imagenet_std.to(x.device)
            x01 = (x * std + mean).clamp(0.0, 1.0)
        else:
            x01 = x.clamp(0.0, 1.0)

        x255 = x01 * 255.0
        x_sam = (x255 - self.model.pixel_mean) / self.model.pixel_std

        # 2) Fix pos_embed alla griglia patch corrente
        patch = enc.patch_embed.proj.stride[0]  # vit_b: 16
        hp, wp = x_sam.shape[-2] // patch, x_sam.shape[-1] // patch

        pos = enc.pos_embed  # (1, H0, W0, C) tipicamente (1,64,64,C)
        if pos.shape[1] != hp or pos.shape[2] != wp:
            pos_ = pos.permute(0, 3, 1, 2)  # (1,C,H0,W0)
            pos_ = F.interpolate(pos_, size=(hp, wp), mode="bilinear", align_corners=False)
            pos_ = pos_.permute(0, 2, 3, 1)  # (1,hp,wp,C)
            enc.pos_embed = nn.Parameter(pos_, requires_grad=False)

        # 3) Encoder features
        feats = enc(x_sam)  # (B, C, hp, wp)
        return feats

    def setup_finetuning(self, num_layers):
        for param in self.model.parameters():
            param.requires_grad = False

        # In SAM i blocchi del ViT sono in self.model.image_encoder.blocks
        blocks = self.model.image_encoder.blocks
        total_blocks = len(blocks)
        for i in range(total_blocks - num_layers, total_blocks):
            for param in blocks[i].parameters():
                param.requires_grad = True

        print(
            f"{self.__class__.__name__}: Sbloccati gli ultimi {num_layers}/{total_blocks} blocchi dell'image_encoder.")