import torch
import torch.nn.functional as F

def compute_similarity_logits(src_feats, trg_feats, src_kps_px, img_h, img_w):
    """
    Parte differenziabile: Calcola la matrice di similarità coseno.
    Utilizzabile sia in Training (con Softmax/Loss) che in Inference (con Argmax).
    
    Args:
        src_feats: (C, Hf, Wf) source feature map
        trg_feats: (C, Hf, Wf) target feature map
        src_kps_px: (Nv, 2) keypoints pixel coordinates [0, img_size]
        img_size: int, dimensione immagine originale (es. 518)
        
    Returns:
        sim: (Nv, Hf*Wf) Matrice di similarità (logits)
    """
    device = src_feats.device
    C, Hf, Wf = src_feats.shape
    Nv = src_kps_px.shape[0]

    if Nv == 0:
        return torch.empty((0, Hf * Wf), device=device)

    # 1) Normalize per-location vectors
    src_feats = F.normalize(src_feats, dim=0)
    trg_feats = F.normalize(trg_feats, dim=0)

    # Stride (necessario per mappare pixel -> feature grid)
    stride_x = img_w / Wf
    stride_y = img_h / Hf

    # 2) Pixel -> feature-grid coords per grid_sample (range [-1, 1])
    # Patch center convention: pixel = (idx + 0.5) * stride
    # idx = pixel / stride - 0.5
    xs = src_kps_px[:, 0].to(device) / stride_x - 0.5
    ys = src_kps_px[:, 1].to(device) / stride_y - 0.5

    # Normalize to [-1, 1] for grid_sample
    xs_norm = (xs / (Wf - 1)) * 2 - 1
    ys_norm = (ys / (Hf - 1)) * 2 - 1

    grid = torch.stack([xs_norm, ys_norm], dim=1).view(1, Nv, 1, 2) # (1, Nv, 1, 2)

    # 3) Bilinear sampling on src features
    # Aggiungi batch dimension fittizia per grid_sample: (1, C, Hf, Wf)
    src_vecs = F.grid_sample(
        src_feats.unsqueeze(0), 
        grid, 
        mode="bilinear", 
        align_corners=True
    ) # Output: (1, C, Nv, 1)

    src_vecs = src_vecs.squeeze(0).squeeze(-1).T  # (Nv, C)

    # Rinormalizza dopo l'interpolazione (importante perché l'interpolazione altera la norma)
    src_vecs = F.normalize(src_vecs, dim=1)

    # 4) Flatten target features
    # trg_feats è (C, Hf, Wf) -> permute -> (Hf, Wf, C) -> reshape -> (Hf*Wf, C)
    trg_flat = trg_feats.permute(1, 2, 0).reshape(-1, C)

    # 5) Matmul: (Nv, C) @ (C, Hf*Wf) -> (Nv, Hf*Wf)
    sim = torch.matmul(src_vecs, trg_flat.T)
    
    return sim


def find_correspondences(src_feats, trg_feats, src_kps_px, img_size):
    """
    Wrapper per l'INFERENZA (Test/Validation).
    Chiama la logica condivisa e applica l'argmax (non differenziabile).
    
    Args:
        src_feats: (C, Hf, Wf) source feature map
        trg_feats: (C, Hf, Wf) target feature map
        src_kps_px: (Nv, 2) source keypoints in pixel coords
        img_size: resized image size
        
    Returns:
        pred_kps_px: (Nv, 2) predicted target keypoints in pixel coords
    """
    # --- 1. Parte Condivisa (Differenziabile) ---
    sim = compute_similarity_logits(src_feats, trg_feats, src_kps_px, img_size)
    
    device = src_feats.device
    Nv = sim.shape[0]
    if Nv == 0:
        return torch.empty((0, 2), device=device)
        
    _, Hf, Wf = trg_feats.shape
    stride_x = img_size / Wf
    stride_y = img_size / Hf

    # --- 2. Parte Specifica Inference (Argmax) ---
    max_idx = sim.argmax(dim=1)

    pred_y = (max_idx // Wf).float()
    pred_x = (max_idx % Wf).float()

    # Feature Grid -> Pixel coords (Patch Center Convention)
    pred_x_px = (pred_x + 0.5) * stride_x
    pred_y_px = (pred_y + 0.5) * stride_y

    pred_x_px = pred_x_px.clamp(0, img_size - 1)
    pred_y_px = pred_y_px.clamp(0, img_size - 1)

    return torch.stack([pred_x_px, pred_y_px], dim=1)
