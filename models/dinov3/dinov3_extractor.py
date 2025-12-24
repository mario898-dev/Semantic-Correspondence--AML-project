import torch
import torch.nn as nn
import sys
import os

class DINOv3Extractor(nn.Module):
    def __init__(self, repo_dir: str, model_name: str, weights: str, device: str = 'cuda'):
        super().__init__()
        self.device = device
        
        # 1. Aggiungiamo la root del modulo (external/dinov3) al path
        abs_repo_dir = os.path.abspath(repo_dir)
        if abs_repo_dir not in sys.path:
            sys.path.insert(0, abs_repo_dir)
        
        # 2. Importiamo il costruttore dal tuo file vision_transformer.py
        try:
            from dinov3.models import vision_transformer as vits
        except ImportError as e:
            raise ImportError(f"Assicurati che {abs_repo_dir} contenga la cartella 'dinov3'. Errore: {e}")

        # 3. Inizializzazione basata sulle funzioni factory del file
        # Usiamo n_storage_tokens=4 (standard DINOv3 per i registri)
        if "vits16" in model_name:
            self.model = vits.vit_small(patch_size=16, n_storage_tokens=4)
        elif "vitb14" in model_name:
            self.model = vits.vit_base(patch_size=14, n_storage_tokens=4)
        else:
            raise ValueError(f"Configurazione non trovata per {model_name}")

        # 4. Caricamento pesi
        if os.path.exists(weights):
            checkpoint = torch.load(weights, map_location='cpu')
            # Il file sorgente suggerisce che i pesi siano sotto 'model'
            state_dict = checkpoint.get("model", checkpoint)
            msg = self.model.load_state_dict(state_dict, strict=True)
            print(f"✅ DINOv3: Pesi caricati correttamente ({msg})")
        else:
            raise FileNotFoundError(f"Pesi non trovati in {weights}")

        self.model.to(device).eval()
        
        # Attributi per eval.py
        self.patch_size = self.model.patch_size
        self.embed_dim = self.model.embed_dim

    @torch.no_grad()
    def forward(self, x):
        """
        Input: (B, 3, H, W)
        Output: (B, embed_dim, H/16, W/16)
        """
        B, C, H, W = x.shape
        h_patches, w_patches = H // self.patch_size, W // self.patch_size
        
        # forward_features nel tuo codice restituisce un Dict:
        # { "x_norm_clstoken": ..., "x_storage_tokens": ..., "x_norm_patchtokens": ... }
        features = self.model.forward_features(x)
        
        # Estraggono i patch tokens (già normalizzati con self.norm)
        # Il codice sorgente li separa automaticamente tramite n_storage_tokens + 1
        patch_features = features['x_norm_patchtokens'] # Shape: (B, N_patches, C)
        
        # Reshape spaziale per il matching
        patch_features = patch_features.transpose(1, 2).reshape(B, self.embed_dim, h_patches, w_patches)
        
        return patch_features
