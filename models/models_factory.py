import os
from .dinov2_extractor import DINOv2Extractor
from .SAM_extractor import SAMExtractor
from .dinov3.dinov3_extractor import DINOv3Extractor

def build_model(backbone: str, device: str):
    if backbone == "dinov2_vitb14":
        return DINOv2Extractor("dinov2_vitb14", device)

    elif backbone == "dinov2_vitl14":
        return DINOv2Extractor("dinov2_vitl14", device)

    elif backbone == "dinov3_vits16":
        # Nota: puoi rendere i path relativi alla REPO_ROOT se necessario
        return DINOv3Extractor(
            repo_dir="external/dinov3",
            model_name="dinov3_vits16",
            weights="checkpoints/DINOv3/dinov3_vits16.pth",
            device=device,
        )

    elif backbone == "sam_vitb":
        return SAMExtractor(
            repo_dir="external/segment-anything",
            model_type="vit_b",
            device=device,
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone}")