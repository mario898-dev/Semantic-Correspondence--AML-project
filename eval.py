import os
import sys
import torch
from tqdm import tqdm
from collections import defaultdict
import wandb

from utils.cli import parse_eval_args
from dataset.spair import SPairDataset
from dataset.pfpascal import PFPascalDataset
from dataset.pfwillow import PFWillowDataset
from utils.matching import find_correspondences
from utils.metrics import compute_pck_metrics
from project_config import Config
from models.models_factory import build_model


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def run_evaluation(args):

    # Questo aggiorna Config.DATASET.IMG_SIZE in base al backbone scelto (es. 'dinov3_vitl16')
    Config.DATASET.set_resolution(args.backbone)

    # --- SETUP DISPOSITIVO E MODELLO ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizzando il dispositivo: {device}")

    model = build_model(args.backbone, device, 0, args.weights)
    model.eval()

    # --- DATASET ---
    if args.dataset == "spair":
        test_dataset = SPairDataset(
            cfg=Config,
            split="test",
            category=args.category
        )
    elif args.dataset == "pfwillow":
        test_dataset = PFWillowDataset(
            cfg=Config, 
            split="test", 
            category=args.category
        )
    elif args.dataset == "pfpascal":
        test_dataset = PFPascalDataset(
            cfg=Config, 
            split="test", 
            category=args.category
        )
    else:
        raise ValueError(f"Dataset {Config.DATASET.NAME} non supportato!")

    print(f"IMG_SIZE usata in eval: {Config.DATASET.IMG_SIZE}")

    # --- METRICHE ---
    alphas = Config.EVALUATOR.ALPHA

    correct_points = defaultdict(int)
    total_points = defaultdict(int)
    pck_images = defaultdict(list)

    correct_points_cat = defaultdict(lambda: defaultdict(int))
    total_points_cat = defaultdict(lambda: defaultdict(int))
    pck_images_cat = defaultdict(lambda: defaultdict(list))

    total_keypoints_valid = 0
    total_keypoints_all = 0
    num_pairs_used = 0

    print(f"\n{'='*60}")
    print(f"VALUTAZIONE TRAINING-FREE su SPair-71k ({len(test_dataset)} pairs)")
    print(f"{'='*60}\n")
    
    print("Using window soft argmax (task3)" if args.use_window_soft == 1 else "Using argmax (task1)")

    # --- WANDB INIT (solo se attivo) ---
    if args.wandb:
        wandb.init(
            project="AML-Semantic-Correspondence",
            name=f"{args.backbone}-{args.category}",
            mode=args.wandb_mode,
            config={
                "backbone": args.backbone,
                "category": args.category,
                "dataset": "SPair-71k",
                "mode": "training-free",
                "pck_thresholds": Config.EVALUATOR.ALPHA,
            },
        )

    # --- LOOP DI VALUTAZIONE ---
    for idx in tqdm(range(len(test_dataset))):
        batch = test_dataset[idx]

        src_img = batch["src_img"].unsqueeze(0).to(device)  # (1,3,H,W)
        trg_img = batch["trg_img"].unsqueeze(0).to(device)

        src_kps = batch["src_kps"]  # (N, 2)
        trg_kps = batch["trg_kps"]  # (N, 2)

        category = batch.get("category", "all")
        pckthres = batch["pckthres"].item()

        # Keypoint validi
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
            forward_kwargs = {}
            
            # Passiamo extract_layer SOLO se è SAM e se l'utente l'ha specificato
            if "sam" in args.backbone and args.extract_layer is not None:
                forward_kwargs["extract_layer"] = args.extract_layer
            
            # Chiamata al modello con unpacking dei kwargs
            # Nota: [0] serve perché il tuo eval si aspetta di rimuovere la dimensione batch
            src_feats = model(src_img, **forward_kwargs)[0]  # (C, Hf, Wf)
            trg_feats = model(trg_img, **forward_kwargs)[0]  # (C, Hf, Wf)

        # Dimensioni reali immagine (coerenti con i keypoint del dataset)
        img_h, img_w = src_img.shape[-2:]

        # Corrispondenze
        pred_kps_valid = find_correspondences(
            src_feats,
            trg_feats,
            src_kps_valid.to(device),
            img_h,
            img_w,
            use_window_soft=args.use_window_soft
        ).cpu()

        # Metriche
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

    # --- RISULTATI FINALI ---
    print(f"\n✅ Valutazione completata!")
    print(f"   Pair usati (>=1 kp valido): {num_pairs_used} / {len(test_dataset)}")
    print(
        f"   Keypoint validi: {total_keypoints_valid} / {total_keypoints_all} "
        f"({100 * total_keypoints_valid / total_keypoints_all:.1f}%)"
    )

    pck_per_point = {
        a: (100.0 * correct_points[a] / total_points[a]) if total_points[a] > 0 else 0.0
        for a in alphas
    }

    pck_per_image = {
        a: (sum(pck_images[a]) / len(pck_images[a])) if len(pck_images[a]) > 0 else 0.0
        for a in alphas
    }

    print("\n--- PCK (dataset) ---")
    for a in alphas:
        print(
            f"PCK@{a:.2f}  "
            f"per-point: {pck_per_point[a]:6.2f}%   "
            f"per-image: {pck_per_image[a]:6.2f}%"
        )

    # --- PCK (per category): stampa sempre, wandb solo se attivo ---
    print("\n--- PCK (per category) ---")

    cat_table = None
    if args.wandb:
        cat_table = wandb.Table(
            columns=[
                "category",
                *[f"PCK@{a:.2f}_per_point" for a in alphas],
                *[f"PCK@{a:.2f}_per_image" for a in alphas],
            ]
        )

    for cat in sorted(pck_images_cat.keys()):
        pp_vals = []
        pi_vals = []
        for a in alphas:
            pp = (
                100.0 * correct_points_cat[cat][a] / total_points_cat[cat][a]
            ) if total_points_cat[cat][a] > 0 else 0.0

            pi = (
                sum(pck_images_cat[cat][a]) / len(pck_images_cat[cat][a])
            ) if len(pck_images_cat[cat][a]) > 0 else 0.0

            pp_vals.append(pp)
            pi_vals.append(pi)

        # stampa sempre
        pretty = [f"{cat:>15}"]
        for i, a in enumerate(alphas):
            pretty.append(f"PCK@{a:.2f} pp {pp_vals[i]:6.2f}% | pi {pi_vals[i]:6.2f}%")
        print("   ".join(pretty))

        # tabella wandb solo se attivo
        if args.wandb:
            cat_table.add_data(cat, *pp_vals, *pi_vals)

    # --- WANDB LOG (solo se attivo) ---
    if args.wandb:
        wandb.log({f"PCK@{a:.2f}_per_point": pck_per_point[a] for a in alphas})
        wandb.log({f"PCK@{a:.2f}_per_image": pck_per_image[a] for a in alphas})
        wandb.log({"PCK_per_category": cat_table})
        wandb.finish()


if __name__ == "__main__":
    args = parse_eval_args()
    run_evaluation(args)






