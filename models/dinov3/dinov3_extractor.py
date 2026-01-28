import torch
import torch.nn as nn
import sys
import os

class DINOv3Extractor(nn.Module):
    def __init__(self, repo_dir: str, model_name: str, weights: str, device: str = 'cuda'):
        super().__init__()
        self.device = device
        
        # 1. Setup path per importare il modulo dinov3 locale
        abs_repo_dir = os.path.abspath(repo_dir)
        if abs_repo_dir not in sys.path:
            sys.path.insert(0, abs_repo_dir)
        
        try:
            from dinov3.models import vision_transformer as vits
        except ImportError as e:
            raise ImportError(f"Controlla che {abs_repo_dir} contenga la cartella 'dinov3'. Errore: {e}")

        # 2. Configurazione per matchare i pesi ufficiali di Meta
        # Questi parametri attivano le componenti che causavano l'errore 'Unexpected key'
        meta_config = {
            "n_storage_tokens": 4,    # I 4 registri di DINOv3
            "layerscale_init": 1e-5,  # Crea i parametri 'gamma' (ls1, ls2)
            "mask_k_bias": True,      # Crea i parametri 'bias_mask'
        }

        # 3. Inizializzazione modello
        if "vits16" in model_name:
            self.model = vits.vit_small(patch_size=16, **meta_config)
        elif "vitb16" in model_name:
            self.model = vits.vit_base(patch_size=16, **meta_config)
        else:
            raise ValueError(f"Modello {model_name} non supportato.")

        # 4. Caricamento pesi con gestione del dizionario Meta
        if os.path.exists(weights):
            try:
                checkpoint = torch.load(weights, map_location='cpu', weights_only=False)
            except TypeError:
                # Fallback per versioni vecchie di torch che non hanno weights_only
                checkpoint = torch.load(weights, map_location='cpu')

            if isinstance(checkpoint, dict) and "model" in checkpoint:
                state_dict = checkpoint["model"]
                print("Rilevata struttura checkpoint con chiave ['model']. Estrazione in corso...")
            else:
                state_dict = checkpoint
            
            # Rimozione del prefisso 'model.' dalle chiavi dello state_dict.
            # Necessario quando il checkpoint e' stato salvato con un wrapper 
            # (es. DataParallel) che aggiunge questo prefisso
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("model."):
                    new_state_dict[k.replace("model.", "", 1)] = v
                else:
                    new_state_dict[k] = v

            msg = self.model.load_state_dict(new_state_dict, strict=True)
            print(f"DINOv3 caricato con successo: {msg}")
            print(f"DINOv3 caricato con successo dal file: {os.path.basename(weights)}")
        else:
            raise FileNotFoundError(f"Pesi non trovati: {weights}")

        self.model.to(device)
        self.patch_size = self.model.patch_size
        self.embed_dim = self.model.embed_dim


    def forward(self, x):
        B, C, H, W = x.shape
        h_patches, w_patches = H // self.patch_size, W // self.patch_size
        
        # Estrazione feature tramite il dizionario restituito da Meta
        features = self.model.forward_features(x)
        
        # 'x_norm_patchtokens' contiene i token spaziali puliti (no CLS, no Registri)
        patch_features = features['x_norm_patchtokens'] 
        
        # Reshape: (B, N, C) -> (B, C, H_p, W_p)
        patch_features = patch_features.transpose(1, 2).reshape(B, self.embed_dim, h_patches, w_patches)
        
        return patch_features

    def setup_finetuning(self, num_layers):
        for param in self.model.parameters():
            param.requires_grad = False

        # In DINOv3, self.model è il ViT che ha l'attributo 'blocks'
        total_blocks = len(self.model.blocks)
        for i in range(total_blocks - num_layers, total_blocks):
            for param in self.model.blocks[i].parameters():
                param.requires_grad = True

        print(f"{self.__class__.__name__}: Sbloccati gli ultimi {num_layers}/{total_blocks} blocchi.")
