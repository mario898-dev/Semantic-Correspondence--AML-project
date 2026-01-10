import torch
import torch.nn as nn
import os

class DINOv2Extractor(nn.Module):
    def __init__(self, model_name: str, device: str = 'cuda', weights: str=None):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model = self.model.to(device)
        if weights is not None and isinstance(weights, str):
            if os.path.exists(weights):
                print(f"📥 Caricamento pesi custom da: {weights}")
                
                # Caricamento sicuro (gestisce compatibilità torch 2.6+)
                try:
                    checkpoint = torch.load(weights, map_location=device, weights_only=False)
                except TypeError:
                    checkpoint = torch.load(weights, map_location=device)

                # A. Gestione se il checkpoint è un dizionario (es. contiene 'epoch', 'model')
                if isinstance(checkpoint, dict) and "model" in checkpoint:
                    state_dict = checkpoint["model"]
                    print("ℹ️ Checkpoint: Trovata chiave 'model'.")
                else:
                    state_dict = checkpoint

                # B. Pulizia Prefissi (Cruciale per il tuo file best.pth)
                # Il tuo file ha chiavi tipo "model.blocks.0...", ma self.model vuole "blocks.0..."
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("model."):
                        # Rimuove il prefisso "model."
                        new_state_dict[k.replace("model.", "", 1)] = v
                    else:
                        new_state_dict[k] = v
                
                # C. Caricamento effettivo
                # strict=False evita crash se mancano pezzi non essenziali (es. la head originale)
                msg = self.model.load_state_dict(new_state_dict, strict=False)
                print(f"✅ Pesi custom caricati! Risultato load: {msg}")
            else:
                print(f"⚠️ ATTENZIONE: File pesi '{weights}' non trovato. Uso i pesi default di DINOv2.")
        self.patch_size = self.model.patch_size
        self.embed_dim = self.model.embed_dim
        self.device = device
        print(f"✅ {model_name} loaded (patch_size={self.patch_size}, dim={self.embed_dim})")


    def forward(self, x):
        B, C, H, W = x.shape
        features = self.model.forward_features(x)
        patch_features = features['x_norm_patchtokens']
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size
        patch_features = patch_features.transpose(1, 2).reshape(B, self.embed_dim, h_patches, w_patches)
        return patch_features

    def setup_finetuning(self, num_layers):
        # 1. Congela tutto il modello
        for param in self.model.parameters():
            param.requires_grad = False

        # 2. Scongela gli ultimi N blocchi
        # In DINOv2, self.model è il ViT che ha l'attributo 'blocks'
        total_blocks = len(self.model.blocks)
        for i in range(total_blocks - num_layers, total_blocks):
            for param in self.model.blocks[i].parameters():
                param.requires_grad = True

        print(f"🔥 {self.__class__.__name__}: Sbloccati gli ultimi {num_layers}/{total_blocks} blocchi.")
