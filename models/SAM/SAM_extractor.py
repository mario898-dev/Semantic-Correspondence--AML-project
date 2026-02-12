import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class SAMExtractor(nn.Module):
    def __init__(self, repo_dir: str, model_type: str = "vit_b", weights: str = None, device: str = "cuda"):
        """
        Wrapper for the Segment Anything (SAM) encoder for Semantic Correspondence.
        Handles weight loading, dynamic positional embedding interpolation, and partial fine-tuning.
        """
        super().__init__()
        self.device = device

        # 1. Setup path to import the external segment-anything library
        abs_repo_dir = os.path.abspath(repo_dir)
        if abs_repo_dir not in sys.path:
            sys.path.insert(0, abs_repo_dir)

        try:
            from segment_anything import sam_model_registry
        except ImportError as e:
            raise ImportError(f"Error importing SAM from {abs_repo_dir}. Make sure the folder exists. Error: {e}")

        # 2. Initialize SAM architecture
        # Create model without automatic weight loading,
        # to manually handle loading in the next section
        print(f"Building SAM architecture ({model_type})...")
        self.model = sam_model_registry[model_type](checkpoint=None)
        self.model.to(device)

        # 3. Load pre-trained model weights
        if weights and os.path.exists(weights):
            print(f"Loading weights from: {os.path.basename(weights)}")
            try:
                # weights_only=False allows loading checkpoints saved in legacy formats
                # Note: use with caution for files from untrusted sources
                checkpoint = torch.load(weights, map_location=device, weights_only=False)
            except TypeError:
                # Fallback for PyTorch versions that don't support weights_only
                checkpoint = torch.load(weights, map_location=device)

            # Extract state_dict: the checkpoint can be a dictionary
            # with metadata (e.g. epoch, optimizer) or directly the state_dict
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint

            # Clean up keys: remove 'model.' prefix that gets added
            # when saving a model wrapped in DataParallel or DistributedDataParallel
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("model."):
                    new_state_dict[k.replace("model.", "", 1)] = v
                else:
                    new_state_dict[k] = v
            
            # Load with strict=False to ignore missing keys
            # due to differences between model versions
            msg = self.model.load_state_dict(new_state_dict, strict=False)
            print(f"SAM weights loaded. Report: {msg}")
        elif weights:
            raise FileNotFoundError(f"Weights file not found: {weights}")
        else:
            print("WARNING: No weights specified. SAM initialized with random weights (useful for debug only).")

        # 4. Normalization buffers (ImageNet mean/std for input preprocessing)
        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, extract_layer: int = None) -> torch.Tensor:
        """
        Forward pass of the image encoder.
        x: (B, 3, H, W) -> Images can have arbitrary size.
        Returns: (B, C, H/16, W/16) -> Dense feature map.
        """
        enc = self.model.image_encoder

        # --- A. Preprocessing ---
        # SAM expects internally normalized input based on 0-255.
        # Here we handle both ImageNet-normalized inputs (typical from Dataloader) and [0,1].
        if x.max() > 2.0 or x.min() < -1.0:
            # Assume ImageNet standardization -> Denormalize to [0,1]
            mean = self.imagenet_mean.to(x.device)
            std = self.imagenet_std.to(x.device)
            x01 = (x * std + mean).clamp(0.0, 1.0)
        else:
            # Assume already in [0,1]
            x01 = x.clamp(0.0, 1.0)

        # Scale to 0-255 and apply SAM-specific normalization
        x255 = x01 * 255.0
        x_sam = (x255 - self.model.pixel_mean) / self.model.pixel_std

        # --- B. Dynamic Positional Embedding Handling ---
        # SAM is trained on 1024x1024. If image is different, interpolate pos_embed.
        # CRUCIAL: Do not overwrite enc.pos_embed, use a temporary variable.
        patch_size = enc.patch_embed.proj.stride[0]
        hp, wp = x_sam.shape[-2] // patch_size, x_sam.shape[-1] // patch_size

        # Reference to original weights
        pos_embed = enc.pos_embed
        
        # Interpolate if patch dimensions don't match native ones (64x64)
        if pos_embed.shape[1] != hp or pos_embed.shape[2] != wp:
            pos_embed = pos_embed.permute(0, 3, 1, 2)  # (1, C, H, W)
            pos_embed = F.interpolate(pos_embed, size=(hp, wp), mode="bilinear", align_corners=False)
            pos_embed = pos_embed.permute(0, 2, 3, 1)  # (1, H, W, C)

        # --- C. Feature Extraction ---
        # 1. Patch Embedding
        out = enc.patch_embed(x_sam)
        
        # 2. Add Positional Embedding (using the interpolated tensor)
        if enc.pos_embed is not None:
            out = out + pos_embed 

        # 3. Forward through Transformer blocks
        limit = len(enc.blocks) if extract_layer is None else extract_layer + 1

        for i, blk in enumerate(enc.blocks):
            out = blk(out)
            if i == limit - 1:
                break
        
        # Output shape: (B, H_patch, W_patch, C) -> Permute to (B, C, H_patch, W_patch)
        feats = out.permute(0, 3, 1, 2)
        return feats

    def setup_finetuning(self, num_layers: int):
        """
        Configure layers for fine-tuning.
        Freeze the entire encoder, then unfreeze the last 'num_layers' blocks.
        """
        # 1. Freeze everything initially
        for param in self.model.image_encoder.parameters():
            param.requires_grad = False

        blocks = self.model.image_encoder.blocks
        total_blocks = len(blocks)
        
        # 2. Unfreeze the last k layers
        if num_layers > 0:
            start_layer = total_blocks - num_layers
            if start_layer < 0: start_layer = 0
            
            for i in range(start_layer, total_blocks):
                for param in blocks[i].parameters():
                    param.requires_grad = True
            
            print(f"{self.__class__.__name__}: Unfroze last {num_layers}/{total_blocks} blocks for fine-tuning.")
        else:
            print(f"{self.__class__.__name__}: Encoder fully frozen (no trainable parameters).")
