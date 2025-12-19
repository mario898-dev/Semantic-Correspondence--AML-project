import torch
import torch.nn.functional as F


def find_correspondences(src_feats, trg_feats, src_kps_px, img_size):
    """
    Training-free correspondence via cosine similarity (patch tokens).
    IMPORTANT:
    - campiona src_feats in modo coerente con "centro patch"
    - rinormalizza i vettori dopo bilinear sampling
    - converte argmax (patch index) -> pixel usando centro patch

    Args:
        src_feats: (C, Hf, Wf) source feature map (patch grid)
        trg_feats: (C, Hf, Wf) target feature map (patch grid)
        src_kps_px: (Nv, 2) VALID source keypoints in pixel coords [0, img_size-1]
        img_size: resized image size (e.g. 518)

    Returns:
        pred_kps_px: (Nv, 2) predicted target keypoints in pixel coords
    """
    device = src_feats.device
    C, Hf, Wf = src_feats.shape
    Nv = src_kps_px.shape[0]

    if Nv == 0:
        return torch.empty((0, 2), device=device)

    # 1) Normalize per-location vectors (cosine similarity)
    # src_feats/trg_feats: normalize across channel dim for each spatial location
    src_feats = F.normalize(src_feats, dim=0)
    trg_feats = F.normalize(trg_feats, dim=0)

    # stride in pixel per patch cell (robusto anche se non combacia perfettamente con patch_size)
    stride_x = img_size / Wf
    stride_y = img_size / Hf

    # 2) Pixel -> feature-grid continuous coords (patch-center convention)
    # Patch j ha centro a (j+0.5)*stride_x
    # Quindi: j = x_px/stride_x - 0.5
    xs = src_kps_px[:, 0].to(device) / stride_x - 0.5  # in [~0..Wf-1]
    ys = src_kps_px[:, 1].to(device) / stride_y - 0.5  # in [~0..Hf-1]

    # feature coords -> normalized coords for grid_sample in [-1, 1]
    xs_norm = (xs / (Wf - 1)) * 2 - 1
    ys_norm = (ys / (Hf - 1)) * 2 - 1

    grid = torch.stack([xs_norm, ys_norm], dim=1).view(1, Nv, 1, 2)  # (1, Nv, 1, 2)

    # 3) Bilinear sampling on src feature map
    src_feats_batched = src_feats.unsqueeze(0)  # (1, C, Hf, Wf)
    src_vecs = F.grid_sample(
        src_feats_batched,
        grid,
        mode="bilinear",
        align_corners=True
    )  # (1, C, Nv, 1)

    src_vecs = src_vecs.squeeze(0).squeeze(-1).T  # (Nv, C)

    # IMPORTANT: bilinear interpolation rompe la norma -> rinormalizza qui
    src_vecs = F.normalize(src_vecs, dim=1)

    # 4) Flatten target feature map (already unit-normalized per-location)
    trg_flat = trg_feats.permute(1, 2, 0).reshape(-1, C)  # (Hf*Wf, C)

    # 5) Cosine similarity + argmax
    sim = torch.matmul(src_vecs, trg_flat.T)  # (Nv, Hf*Wf)
    max_idx = sim.argmax(dim=1)

    pred_y = (max_idx // Wf).float()
    pred_x = (max_idx % Wf).float()

    # 6) Patch-grid -> pixel coords using PATCH CENTER
    pred_x_px = (pred_x + 0.5) * stride_x
    pred_y_px = (pred_y + 0.5) * stride_y

    pred_x_px = pred_x_px.clamp(0, img_size - 1)
    pred_y_px = pred_y_px.clamp(0, img_size - 1)

    return torch.stack([pred_x_px, pred_y_px], dim=1)
