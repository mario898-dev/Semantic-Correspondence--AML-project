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
                print(f"Loading custom weights from: {weights}")
                
                # Safe checkpoint loading, compatible with PyTorch 2.6+
                try:
                    checkpoint = torch.load(weights, map_location=device, weights_only=False)
                except TypeError:
                    checkpoint = torch.load(weights, map_location=device)

                # Extract state_dict if checkpoint contains additional metadata
                if isinstance(checkpoint, dict) and "model" in checkpoint:
                    state_dict = checkpoint["model"]
                    print("Checkpoint: Detected structure with 'model' key.")
                else:
                    state_dict = checkpoint

                # Remove 'model.' prefix from state_dict keys.
                # Necessary when checkpoint was saved with a wrapper
                # like DataParallel, which prepends 'model.' to all keys
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("model."):
                        new_state_dict[k.replace("model.", "", 1)] = v
                    else:
                        new_state_dict[k] = v
                
                # Load with strict=False to tolerate differences
                # between model structure and saved weights
                msg = self.model.load_state_dict(new_state_dict, strict=False)
                print(f"Custom weights loaded. Result: {msg}")
            else:
                print(f"WARNING: Weights file '{weights}' not found. Using default DINOv2 weights.")
        self.patch_size = self.model.patch_size
        self.embed_dim = self.model.embed_dim
        self.device = device
        print(f"{model_name} loaded (patch_size={self.patch_size}, dim={self.embed_dim})")


    def forward(self, x):
        B, C, H, W = x.shape
        features = self.model.forward_features(x)
        patch_features = features['x_norm_patchtokens']
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size
        patch_features = patch_features.transpose(1, 2).reshape(B, self.embed_dim, h_patches, w_patches)
        return patch_features

    def setup_finetuning(self, num_layers):
        # 1. Freeze the entire model
        for param in self.model.parameters():
            param.requires_grad = False

        # 2. Unfreeze the last N blocks
        # In DINOv2, self.model is the ViT with the 'blocks' attribute
        total_blocks = len(self.model.blocks)
        for i in range(total_blocks - num_layers, total_blocks):
            for param in self.model.blocks[i].parameters():
                param.requires_grad = True

        print(f"{self.__class__.__name__}: Unfroze last {num_layers}/{total_blocks} blocks for fine-tuning.")
