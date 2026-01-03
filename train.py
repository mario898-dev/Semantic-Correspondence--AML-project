import os
import sys
import torch
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader

# --- Setup Path ---
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SD4MATCH_ROOT = os.path.join(REPO_ROOT, "external", "SD4Match")
sys.path.insert(0, SD4MATCH_ROOT)

# Imports del progetto
from project_config import Config
from dataset.spair import SPairDataset
from models.models_factory import build_model
from project_utils.cli import parse_train_args
from project_utils.matching import compute_similarity_logits
from project_utils.loss import WindowSoftTargetLoss

def run_training(args):
    # 1. Setup Dispositivo e Modello
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device training: {device}")
    
    print(f"Costruzione modello {args.backbone} con {args.trainable_layers} layer addestrabili...")
    # build_model gestisce il sblocco dei layer tramite num_trainable_layers
    model = build_model(args.backbone, device, num_trainable_layers=args.trainable_layers)
    
    # Importante: model.train() abilita il calcolo dei gradienti e Dropout/BatchNorm (se presenti)
    model.train()

    # 2. Dataset e DataLoader
    print(f"Caricamento SPair-71k (split: train, category: {args.category})...")
    train_dataset = SPairDataset(
        cfg=Config,
        split="trn",
        category=args.category
    )
    
    # Utilizziamo batch_size=1 perché il numero di keypoint varia per ogni coppia
    # e il codice esistente in SPairDataset/matching non gestisce batching complesso di keypoints.
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=None 
    )
    
    print(f"Training su {len(train_dataset)} coppie.")

    # 3. Optimizer e Loss
    # Ottimizziamo SOLO i parametri che hanno requires_grad=True (quelli sbloccati)
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params_to_optimize, lr=args.lr)
    
    criterion = WindowSoftTargetLoss(sigma=args.sigma, temperature=0.1)
    
    print(f"Parametri da ottimizzare: {len(params_to_optimize)} tensor(s).")

    # 4. WandB Init
    if args.wandb:
        wandb.init(
            project="AML-Semantic-Correspondence",
            name=f"TRAIN-{args.backbone}-{args.category}-L{args.trainable_layers}",
            mode=args.wandb_mode,
            config=vars(args)
        )

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)
    img_size = Config.DATASET.IMG_SIZE

    # 5. Training Loop
    global_step = 0
    
    for epoch in range(args.epochs):
        print(f"\n{'='*20} Epoch {epoch+1}/{args.epochs} {'='*20}")
        epoch_loss = 0.0
        num_batches = 0
        
        model.train()
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}")
        
        for batch in pbar:
            src_img = batch["src_img"].to(device) # (B, 3, H, W)
            trg_img = batch["trg_img"].to(device)
            
            src_kps = batch["src_kps"].to(device)
            trg_kps = batch["trg_kps"].to(device)

            # Se batch_size=1, rimuoviamo la dimensione batch extra dai keypoints se necessario
            if src_kps.dim() == 3 and src_kps.shape[0] == 1:
                src_kps = src_kps.squeeze(0)
                trg_kps = trg_kps.squeeze(0)
            
            # Filtro keypoint validi (coordinate >= 0)
            valid_mask = (
                (src_kps[:, 0] >= 0) & (src_kps[:, 1] >= 0) &
                (trg_kps[:, 0] >= 0) & (trg_kps[:, 1] >= 0)
            )
            src_kps_valid = src_kps[valid_mask]
            trg_kps_valid = trg_kps[valid_mask]

            # Se non ci sono keypoint validi in questa coppia, saltiamo
            if len(src_kps_valid) == 0:
                continue

            # --- Forward Pass ---
            optimizer.zero_grad()
            
            # Estrazione feature (nota: model(...) ritorna lista/tupla, prendiamo [0])
            src_feats = model(src_img)[0]
            trg_feats = model(trg_img)[0]
            
            # Rimuovi dimensione batch dalle features se presente (B=1 -> C,H,W)
            if src_feats.dim() == 4:
                src_feats = src_feats.squeeze(0)
                trg_feats = trg_feats.squeeze(0)

            # Calcolo logits di similarità (differenziabile)
            # sim_logits: (N_kps_valid, Hf*Wf)
            sim_logits = compute_similarity_logits(
                src_feats,
                trg_feats,
                src_kps_valid,
                img_size
            )

            # --- Calcolo Loss ---
            # Passiamo le dimensioni della feature map per mappare i pixel GT sulla griglia
            feature_shape = src_feats.shape[-2:] # (Hf, Wf)
            loss = criterion(sim_logits, trg_kps_valid, img_size, feature_shape)

            # --- Backward & Step ---
            loss.backward()
            optimizer.step()

            # --- Logging ---
            loss_val = loss.item()
            epoch_loss += loss_val
            num_batches += 1
            global_step += 1
            
            pbar.set_postfix({"loss": f"{loss_val:.4f}"})
            
            if args.wandb:
                wandb.log({"train_loss": loss_val, "global_step": global_step})

        # Fine Epoca
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        print(f"Fine Epoca {epoch+1} - Avg Loss: {avg_loss:.4f}")
        
        # Salvataggio Checkpoint
        save_name = f"{args.backbone}_{args.category}_ep{epoch+1}.pth"
        save_path = os.path.join(args.output_dir, save_name)
        torch.save(model.state_dict(), save_path)
        print(f"Checkpoint salvato in: {save_path}")

    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    args = parse_train_args()
    run_training(args)


