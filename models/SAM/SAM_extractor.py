import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class SAMExtractor(nn.Module):
    def __init__(self, repo_dir: str, model_type: str = "vit_b", weights: str="", device: str = "cuda"):
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

        # Checkpoint (stesso stile di DINOv3: path passato dall'esterno)
        if os.path.exists(weights):
            ckpt = weights
        else:
            raise FileNotFoundError(f"Pesi SAM non trovati: {weights}")

        # Init SAM
        sam = sam_model_registry[model_type](checkpoint=ckpt)
        self.model = sam.to(device)

        # Costanti ImageNet (buffer)
        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        print(f"✅ SAM loaded: {model_type} (checkpoint: {os.path.basename(ckpt)})")

    def forward(self, x: torch.Tensor, extract_layer: int = None) -> torch.Tensor:
        enc = self.model.image_encoder

        # 1) Input preprocessing per SAM
        if x.max() > 2.0 or x.min() < -1.0:
            mean = self.imagenet_mean.to(x.device)
            std = self.imagenet_std.to(x.device)
            x01 = (x * std + mean).clamp(0.0, 1.0)
        else:
            x01 = x.clamp(0.0, 1.0)

        x255 = x01 * 255.0
        x_sam = (x255 - self.model.pixel_mean) / self.model.pixel_std

        # 2) Fix pos_embed alla griglia patch corrente
        patch = enc.patch_embed.proj.stride[0]
        hp, wp = x_sam.shape[-2] // patch, x_sam.shape[-1] // patch

        pos = enc.pos_embed
        if pos.shape[1] != hp or pos.shape[2] != wp:
            pos_ = pos.permute(0, 3, 1, 2)
            pos_ = F.interpolate(pos_, size=(hp, wp), mode="bilinear", align_corners=False)
            pos_ = pos_.permute(0, 2, 3, 1)
            enc.pos_embed = nn.Parameter(pos_, requires_grad=False)

        if extract_layer is None:
            # CASO A: Nessun layer specificato -> Output finale standard (con Neck, 256 canali)
            # Chiamare enc(x_sam) usa il pos_embed che abbiamo appena aggiornato sopra
            feats = enc(x_sam)
            return feats
        
        else:
            # CASO B: Layer specifico -> Estrazione manuale intermedia (es. 768 canali)
            
            # A. Patch Embedding
            out = enc.patch_embed(x_sam)
            
            # B. Add Positional Embedding
            if enc.pos_embed is not None:
                out = out + enc.pos_embed

            # C. Ciclo sui blocchi fino al layer desiderato
            for i, blk in enumerate(enc.blocks):
                out = blk(out)
                if i == extract_layer:
                    break
            
            # D. Permutazione (B, H, W, C) -> (B, C, H, W)
            feats = out.permute(0, 3, 1, 2)
            return feats

    def setup_finetuning(self, num_layers):
        for param in self.model.parameters():
            param.requires_grad = False

        blocks = self.model.image_encoder.blocks
        total_blocks = len(blocks)
        for i in range(total_blocks - num_layers, total_blocks):
            for param in blocks[i].parameters():
                param.requires_grad = True

        print(f"{self.__class__.__name__}: Sbloccati gli ultimi {num_layers}/{total_blocks} blocchi dell'image_encoder.")
