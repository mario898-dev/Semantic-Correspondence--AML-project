import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import sys
sys.path.insert(0, "external/SD4Match")

from data.spair import SPairDataset
from models.dinov2_extractor import DINOv2Extractor
from utils.matching import find_correspondences
from utils.metrics import compute_pck_metrics
from config import cfg  # Assicurati di avere il file config.py


def run_evaluation():
    # 1. SETUP DISPOSITIVO E MODELLO
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilizzando il dispositivo: {device}")

    model = DINOv2Extractor(model_type='dinov2_vits14', device=device)

    # 2. CARICAMENTO DATASET
    test_dataset = SPairDataset(
        data_path='external/SD4Match/asset/SPair-71k',
        split='test',
        img_size=cfg.DATASET.IMG_SIZE
    )

    # 3. INIZIALIZZAZIONE ACCUMULATORI PER METRICHE
    alphas = cfg.EVALUATOR.ALPHA
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

    # 4. LOOP DI VALUTAZIONE
    for idx in tqdm(range(len(test_dataset))):
        batch = test_dataset[idx]

        src_img = batch["src_img"].unsqueeze(0).to(device)
        trg_img = batch["trg_img"].unsqueeze(0).to(device)
        src_kps = batch["src_kps"]  # (N, 2)
        trg_kps = batch["trg_kps"]  # (N, 2)
        category = batch.get("category", "all")
        img_size = cfg.DATASET.IMG_SIZE
        pckthres = batch["pckthres"].item()

        # FIX: validi solo se presenti in ENTRAMBI (src e trg)
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

        # Estrazione feature
        with torch.no_grad():
            # Prendiamo il primo elemento perché il modello restituisce [Batch, C, H, W]
            src_feats = model(src_img)[0]
            trg_feats = model(trg_img)[0]

        # Calcolo corrispondenze
        pred_kps_valid = find_correspondences(
            src_feats,
            trg_feats,
            src_kps_valid.to(device),
            img_size
        ).cpu()

        # Aggiornamento metriche per ogni alpha (es. 0.01, 0.05, 0.1)
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

    # 5. STAMPA DEI RISULTATI FINALI
    print(f"\n✅ Valutazione completata!")
    print(f"   Pair usati (>=1 kp valido): {num_pairs_used} / {len(test_dataset)}")
    print(f"   Keypoint validi: {total_keypoints_valid} / {total_keypoints_all} "
          f"({100 * total_keypoints_valid / total_keypoints_all:.1f}%)")

    pck_per_point = {
        a: (100.0 * correct_points[a] / total_points[a]) if total_points[a] > 0 else 0.0
        for a in alphas
    }

    print("\n--- RISULTATI PCK TOTALI ---")
    for a in alphas:
        print(f"PCK@{a:.2f}: {pck_per_point[a]:6.2f}%")

    # Stampa per categoria
    print("\n--- PCK PER CATEGORIA ---")
    for cat in sorted(pck_images_cat.keys()):
        row = [f"{cat:>15}"]
        for a in alphas:
            pp = (100.0 * correct_points_cat[cat][a] / total_points_cat[cat][a]) \
                 if total_points_cat[cat][a] > 0 else 0.0
            row.append(f"PCK@{a:.2f}: {pp:6.2f}%")
        print(" | ".join(row))


if __name__ == "__main__":
    run_evaluation()
