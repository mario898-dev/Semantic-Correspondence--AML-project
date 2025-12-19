import torch


def compute_pck_metrics(pred_kps, gt_kps, pckthres, alpha=0.1):
    """
    Compute PCK metrics for a single image, supporting both:
    - PCK per point (to be aggregated globally)
    - PCK per image (to be averaged across images)

    Args:
        pred_kps: (Nv, 2) predicted keypoints in pixel coords
        gt_kps:   (Nv, 2) ground-truth keypoints in pixel coords
        pckthres: normalization factor from the dataset
        alpha:    PCK threshold (e.g. 0.05, 0.10, 0.15)

    Returns:
        num_correct: number of correctly predicted keypoints (for PCK per point)
        num_total:   number of valid keypoints (for PCK per point)
        pck_image:   PCK value (%) for this image (for PCK per image)
    """
    num_total = gt_kps.shape[0]

    if num_total == 0:
        # No valid keypoints in this image
        return 0, 0, None

    # Euclidean distance in pixel space
    dist = torch.norm(pred_kps.float() - gt_kps.float(), dim=1)

    threshold = alpha * pckthres

    correct_mask = dist <= threshold
    num_correct = correct_mask.sum().item()

    # PCK per image = percentage of correct keypoints in THIS image
    pck_image = num_correct / num_total * 100.0

    return num_correct, num_total, pck_image
