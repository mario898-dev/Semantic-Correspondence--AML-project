import torch
import torch.nn as nn

class DINOv2Extractor(nn.Module):
    def __init__(self, model_name: str, device: str = 'cuda', weights: str=None):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model = self.model.to(device)
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
