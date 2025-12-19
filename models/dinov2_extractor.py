class DINOv2Extractor(nn.Module):
    def __init__(self, model_name='dinov2_vitb14', device='cuda'):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model = self.model.to(device).eval()
        self.patch_size = self.model.patch_size
        self.embed_dim = self.model.embed_dim
        self.device = device
        print(f"✅ {model_name} loaded (patch_size={self.patch_size}, dim={self.embed_dim})")

    @torch.no_grad()
    def forward(self, x):
        B, C, H, W = x.shape
        features = self.model.forward_features(x)
        patch_features = features['x_norm_patchtokens']
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size
        patch_features = patch_features.transpose(1, 2).reshape(B, self.embed_dim, h_patches, w_patches)
        return patch_features
