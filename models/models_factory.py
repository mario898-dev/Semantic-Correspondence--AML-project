import os
from .dinov2.dinov2_extractor import DINOv2Extractor
from .SAM.SAM_extractor import SAMExtractor
from .dinov3.dinov3_extractor import DINOv3Extractor

def build_model(backbone: str, device: str, num_trainable_layers:int =0):
    if backbone == "dinov2_vits14":
        model = DINOv2Extractor("dinov2_vits14", device)
    elif backbone == "dinov2_vitb14":
        model =  DINOv2Extractor("dinov2_vitb14", device)

    elif backbone == "dinov2_vitl14":
        model =  DINOv2Extractor("dinov2_vitl14", device)

    elif backbone == "dinov3_vits16":
        # Nota: puoi rendere i path relativi alla REPO_ROOT se necessario
        model =  DINOv3Extractor(
            repo_dir="external/dinov3",
            model_name="dinov3_vits16",
            weights="checkpoints/DINOv3/dinov3_vits16.pth",
            device=device,
        )
     elif backbone == "dinov3_vitb16":
        # Nota: puoi rendere i path relativi alla REPO_ROOT se necessario
        model =  DINOv3Extractor(
            repo_dir="external/dinov3",
            model_name="dinov3_vitb16",
            weights="checkpoints/DINOv3/dinov3_vitb16.pth",
            device=device,
        )

     elif backbone == "sam_vitb":
        model =  SAMExtractor(
            repo_dir="external/segment-anything",
            model_type="vit_b",
            weights="checkpoints/SAM/sam_vit_b.pth",
            device=device,
        )
     else:
        raise ValueError(f"Unknown backbone: {backbone}")

     if num_trainable_layers > 0:
        if hasattr(model, 'setup_finetuning'):
            model.setup_finetuning(num_trainable_layers)
        else:
            print(f"⚠️ Warning: {backbone} non ha ancora un metodo setup_finetuning.")
     return model
