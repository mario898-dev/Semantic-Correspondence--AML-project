import torch
from tqdm import tqdm
from dataset.spair import SPairDataset
from utils.matching import find_correspondences
from utils.metrics import compute_pck_metrics
from project_config import Config

def validate_epoch(model, device, category):
    """
    Run a FULL VALIDATION on the entire dataset.
    Uses split='val' if available, otherwise falls back to 'test'.
    Returns the average PCK.
    """
    model.eval()
    
    # 1. Split selection
    # Try to load 'val'. If it fails (file not found), use 'test'.
    val_split = "val"
    try:
        # Quick initialization test
        _ = SPairDataset(cfg=Config, split=val_split, category=category)
    except Exception:
        print(f"Split '{val_split}' not found or empty. Using 'test' for validation.")
        val_split = "test"

    # 2. Load dataset
    val_dataset = SPairDataset(cfg=Config, split=val_split, category=category)
    num_pairs = len(val_dataset)

    print(f" >> [Full Validation] Running on {num_pairs} pairs (Split: {val_split})...")
    
    target_alpha = 0.10
    total_correct = 0
    total_points = 0
    
    # Progress bar
    pbar = tqdm(range(num_pairs), desc="Validating", leave=False)

    with torch.no_grad():
        for idx in pbar:
            batch = val_dataset[idx]
            
            src_img = batch["src_img"].unsqueeze(0).to(device)
            trg_img = batch["trg_img"].unsqueeze(0).to(device)
            src_kps = batch["src_kps"]
            trg_kps = batch["trg_kps"]
            pckthres = batch["pckthres"].item()

            # Validity mask (>= 0 because padding is -2)
            valid_mask = (src_kps[:, 0] >= 0) & (src_kps[:, 1] >= 0) & \
                         (trg_kps[:, 0] >= 0) & (trg_kps[:, 1] >= 0)
            
            src_kps_valid = src_kps[valid_mask].to(device)
            trg_kps_valid = trg_kps[valid_mask]
            
            # Skip if no valid points
            if len(src_kps_valid) == 0:
                continue

            # Forward pass
            src_feats = model(src_img)[0]
            trg_feats = model(trg_img)[0]
            
            # Find correspondences
            img_h, img_w = src_img.shape[-2:]
            pred_kps = find_correspondences(src_feats, trg_feats, src_kps_valid, img_h, img_w).cpu()
            
            # Metrics
            num_correct, num_total, _ = compute_pck_metrics(
                pred_kps, trg_kps_valid, pckthres, alpha=target_alpha
            )
            
            total_correct += num_correct
            total_points += num_total

    # Final PCK computation
    avg_pck = (total_correct / total_points * 100.0) if total_points > 0 else 0.0
    print(f" >> Full Val PCK@{target_alpha}: {avg_pck:.2f}%")
    
    return avg_pck
