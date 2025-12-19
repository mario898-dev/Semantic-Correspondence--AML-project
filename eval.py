import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import pandas as pd

# ============================================================
# Path robusti (indipendenti dalla working directory)
# ============================================================
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- SD4Match (submodule) ---
SD4MATCH_ROOT = os.path.join(REPO_ROOT, "external", "SD4Match")
sys.path.insert(0, SD4MATCH_ROOT)

from dataset.spair import SPairDataset

# --- TUO CODICE ---
from models.dinov2_extractor import DINOv2Extractor
from project_utils.matching import find_correspondences
from project_utils.metrics import compute_pck_metrics
from project_config import Config 


def run_evaluation():

    # ========================================================
    # 1. SETUP DISPOSITIVO E MODELLO
    # ========================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizzando il dispositivo: {device}")

    model = DINOv2Extractor(
        model_name="dinov2_vits14",
        device=device
    )

    # ========================================================
    # 2. CARICAMENTO DATASET (SD4Match)
    # ========================================================
    test_dataset = SPairDataset(
    split="test",
    thres="bbox",
    img_size=Config.DATASET.IMG_SIZE)

    # ========================================================
    # 3. INIZIALIZZAZIONE METRICHE
    # ========================================================
    alphas = Config.EVALUATOR.ALPHA

    correct_points = defaultdict(int)
    total_points = defaultdict(int)
    pck_images = defaultdict(list)

    correct_points_cat = defaultdict(lambda: defaultdict(int))
    total_points_cat   = defaultdict(lambda: defaultdict(int))
    pck_images_cat     = defaultdict(lambda: defaultdict(list))

    total_keypoints_valid = 0
    total_keypoints_all = 0
    num_pairs_used = 0

    print(f"\n{'='*60}")
    print(f"VALUTAZIONE TRAINING-FREE su SPair-71k ({len(test_dataset)} pairs)")
    print(f"{'='*60}\n")

    # ========================================================
    # 4. LOOP DI VALUTAZIONE
    # ========================================================
    for idx in tqdm(range(len(test_dataset))):

        batch = test_dataset[idx]

        src_img = batch["src_img"].unsqueeze(0).to(device)
        trg_img = batch["trg_img"].unsqueeze(0).to(device)

        src_kps = batch["src_kps"]        # (N, 2)
        trg_kps = batch["trg_kps"]        # (N, 2)

        category = batch.get("category", "all")
        img_size = Config.DATASET.IMG_SIZE
        pckthres = batch["pckthres"].item()

        # ✅ keypoint validi solo se presenti in ENTRAMBI
        valid_mask = (
            (src_kps[:, 0] >= 0) & (src_kps[:, 1] >= 0) &
            (trg_kps[:, 0] >= 0) & (trg_kps[:, 1] >= 0)
        )

        src_kps_valid = src_kps[valid_mask]
        trg_kps_valid = trg_kps[valid_mask]

        total_keypoints_valid += len(src_kps_valid)
        total_keypoints_all += len(src_kps)

        if len(src_kps_valid) == 0:
            continue

        num_pairs_used += 1

        # ----------------------------------------------------
        # Estrazione feature
        # ----------------------------------------------------
        with torch.no_grad():
            src_feats = model(src_img)[0]  # (C, Hf, Wf)
            trg_feats = model(trg_img)[0]  # (C, Hf, Wf)

        # ----------------------------------------------------
        # Corrispondenze
        # ----------------------------------------------------
        pred_kps_valid = find_correspondences(
            src_feats,
            trg_feats,
            src_kps_valid.to(device),
            img_size
        ).cpu()

        # ----------------------------------------------------
        # Metriche
        # ----------------------------------------------------
        for alpha in alphas:
            num_correct, num_total, pck_img = compute_pck_metrics(
                pred_kps_valid,
                trg_kps_valid,
                pckthres,
                alpha=alpha
            )

            correct_points[alpha] += num_correct
            total_points[alpha] += num_total

            correct_points_cat[category][alpha] += num_correct
            total_points_cat[category][alpha] += num_total

            if pck_img is not None:
                pck_images[alpha].append(pck_img)
                pck_images_cat[category][alpha].append(pck_img)

    # ========================================================
    # 5. RISULTATI FINALI
    # ========================================================
    print(f"\n✅ Valutazione completata!")
    print(
        f"   Pair usati (>=1 kp valido): "
        f"{num_pairs_used} / {len(test_dataset)}"
    )
    print(
        f"   Keypoint validi: {total_keypoints_valid} / "
        f"{total_keypoints_all} "
        f"({100 * total_keypoints_valid / total_keypoints_all:.1f}%)"
    )

    pck_per_point = {
        a: (100.0 * correct_points[a] / total_points[a])
        if total_points[a] > 0 else 0.0
        for a in alphas
    }

    pck_per_image = {
        a: (sum(pck_images[a]) / len(pck_images[a]))
        if len(pck_images[a]) > 0 else 0.0
        for a in alphas
    }

    print("\n--- PCK (dataset) ---")
    for a in alphas:
        print(
            f"PCK@{a:.2f}  "
            f"per-point: {pck_per_point[a]:6.2f}%   "
            f"per-image: {pck_per_image[a]:6.2f}%"
        )

    print("\n--- PCK (per category) ---")
    for cat in sorted(pck_images_cat.keys()):
        row = [f"{cat:>15}"]
        for a in alphas:
            pp = (
                100.0 * correct_points_cat[cat][a] /
                total_points_cat[cat][a]
            ) if total_points_cat[cat][a] > 0 else 0.0

            pi = (
                sum(pck_images_cat[cat][a]) /
                len(pck_images_cat[cat][a])
            ) if len(pck_images_cat[cat][a]) > 0 else 0.0

            row.append(f"PCK@{a:.2f} pp {pp:6.2f}% | pi {pi:6.2f}%")

        print("   ".join(row))


if __name__ == "__main__":
    run_evaluation()
