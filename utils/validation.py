import torch
from tqdm import tqdm
from dataset.spair import SPairDataset
from utils.matching import find_correspondences
from utils.metrics import compute_pck_metrics
from project_config import Config

def validate_epoch(model, device, category):
    """
    Esegue una VALIDAZIONE COMPLETA su tutto il dataset.
    Usa split='val' se disponibile, altrimenti fallback su 'test'.
    Restituisce la PCK media.
    """
    model.eval()
    
    # 1. Selezione Split
    # Proviamo a caricare 'val'. Se fallisce (file non trovato), usiamo 'test'.
    val_split = "val"
    try:
        # Test rapido di inizializzazione
        _ = SPairDataset(cfg=Config, split=val_split, category=category)
    except Exception:
        print(f"⚠️ Split '{val_split}' non trovato o vuoto. Uso 'test' per la validazione.")
        val_split = "test"

    # 2. Caricamento Dataset
    val_dataset = SPairDataset(cfg=Config, split=val_split, category=category)
    num_pairs = len(val_dataset)

    print(f" >> [Validazione Completa] Avvio su {num_pairs} coppie (Split: {val_split})...")
    
    target_alpha = 0.10  # Soglia PCK di riferimento
    total_correct = 0
    total_points = 0
    
    # Barra di progresso per monitorare l'avanzamento
    pbar = tqdm(range(num_pairs), desc="Validating", leave=False)

    with torch.no_grad():
        for idx in pbar:
            batch = val_dataset[idx]
            
            src_img = batch["src_img"].unsqueeze(0).to(device)
            trg_img = batch["trg_img"].unsqueeze(0).to(device)
            src_kps = batch["src_kps"]
            trg_kps = batch["trg_kps"]
            pckthres = batch["pckthres"].item()

            # Maschera validità (>= 0 perché il padding è -2)
            valid_mask = (src_kps[:, 0] >= 0) & (src_kps[:, 1] >= 0) & \
                         (trg_kps[:, 0] >= 0) & (trg_kps[:, 1] >= 0)
            
            src_kps_valid = src_kps[valid_mask].to(device)
            trg_kps_valid = trg_kps[valid_mask]
            
            # Se la coppia non ha punti validi, saltiamo
            if len(src_kps_valid) == 0:
                continue

            # Forward
            src_feats = model(src_img)[0]
            trg_feats = model(trg_img)[0]
            
            # Matching
            img_h, img_w = src_img.shape[-2:]
            pred_kps = find_correspondences(src_feats, trg_feats, src_kps_valid, img_h, img_w).cpu()
            
            # Metriche
            num_correct, num_total, _ = compute_pck_metrics(
                pred_kps, trg_kps_valid, pckthres, alpha=target_alpha
            )
            
            total_correct += num_correct
            total_points += num_total

    # Calcolo PCK finale
    avg_pck = (total_correct / total_points * 100.0) if total_points > 0 else 0.0
    print(f" >> Full Val PCK@{target_alpha}: {avg_pck:.2f}%")
    
    return avg_pck
