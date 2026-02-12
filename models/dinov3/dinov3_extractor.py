import torch
import torch.nn as nn
import sys
import os

class DINOv3Extractor(nn.Module):
    def __init__(self, repo_dir: str, model_name: str, weights: str, device: str = 'cuda'):
        super().__init__()
        self.device = device
        
        # 1. Setup path to import the local dinov3 module
        abs_repo_dir = os.path.abspath(repo_dir)
        if abs_repo_dir not in sys.path:
            sys.path.insert(0, abs_repo_dir)
        
        try:
            from dinov3.models import vision_transformer as vits
        except ImportError as e:
            raise ImportError(f"Check that {abs_repo_dir} contains the 'dinov3' folder. Error: {e}")

        # 2. Configuration to match Meta's official weights
        # These parameters activate the components that caused the 'Unexpected key' error
        meta_config = {
            "n_storage_tokens": 4,
            "layerscale_init": 1e-5,
            "mask_k_bias": True,
        }

        # 3. Model initialization
        if "vits16" in model_name:
            self.model = vits.vit_small(patch_size=16, **meta_config)
        elif "vitb16" in model_name:
            self.model = vits.vit_base(patch_size=16, **meta_config)
        else:
            raise ValueError(f"Model {model_name} not supported.")

        # 4. Load weights with Meta dictionary handling
        if os.path.exists(weights):
            try:
                checkpoint = torch.load(weights, map_location='cpu', weights_only=False)
            except TypeError:
                # Fallback for older torch versions that don't support weights_only
                checkpoint = torch.load(weights, map_location='cpu')

            if isinstance(checkpoint, dict) and "model" in checkpoint:
                state_dict = checkpoint["model"]
                print("Detected checkpoint structure with ['model'] key. Extracting...")
            else:
                state_dict = checkpoint
            
            # Remove 'model.' prefix from state_dict keys.
            # Necessary when checkpoint was saved with a wrapper
            # (e.g. DataParallel) that adds this prefix
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("model."):
                    new_state_dict[k.replace("model.", "", 1)] = v
                else:
                    new_state_dict[k] = v

            msg = self.model.load_state_dict(new_state_dict, strict=True)
            print(f"DINOv3 loaded successfully: {msg}")
            print(f"DINOv3 loaded successfully from file: {os.path.basename(weights)}")
        else:
            raise FileNotFoundError(f"Weights not found: {weights}")

        self.model.to(device)
        self.patch_size = self.model.patch_size
        self.embed_dim = self.model.embed_dim


    def forward(self, x):
        B, C, H, W = x.shape
        h_patches, w_patches = H // self.patch_size, W // self.patch_size
        
        # Feature extraction via the dictionary returned by Meta
        features = self.model.forward_features(x)
        
        # 'x_norm_patchtokens' contains the clean spatial tokens (no CLS, no Registers)
        patch_features = features['x_norm_patchtokens'] 
        
        # Reshape: (B, N, C) -> (B, C, H_p, W_p)
        patch_features = patch_features.transpose(1, 2).reshape(B, self.embed_dim, h_patches, w_patches)
        
        return patch_features

    def setup_finetuning(self, num_layers):
        for param in self.model.parameters():
            param.requires_grad = False

        # In DINOv3, self.model is the ViT with the 'blocks' attribute
        total_blocks = len(self.model.blocks)
        for i in range(total_blocks - num_layers, total_blocks):
            for param in self.model.blocks[i].parameters():
                param.requires_grad = True

        print(f"{self.__class__.__name__}: Unfroze last {num_layers}/{total_blocks} blocks.")
