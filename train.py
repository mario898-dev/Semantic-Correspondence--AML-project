import os
import sys
import torch
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader

# --- Setup Path ---
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

from project_config import Config
from dataset.spair import SPairDataset
from models.models_factory import build_model
from utils.cli import parse_train_args
from utils.matching import compute_similarity_logits
from utils.loss import FeatMapLoss

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
    
    criterion = FeatMapLoss()
    
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

            # Filtro keypoint validi (coordinate >= 0)
            valid_mask = (
                (src_kps[:, 0] >= 0) & (src_kps[:, 1] >= 0) &
                (trg_kps[:, 0] >= 0) & (trg_kps[:, 1] >= 0)
            )

            npts = valid_mask.sum(dim=1) # (B,) tensore con il numero di punti validi per immagine

            # Se nessun punto è valido nel batch, saltiamo
            if npts.sum() == 0:
                continue

            # --- Forward Pass ---
            optimizer.zero_grad()
            
            # Estrazione feature
            out_src = model(src_img)
            out_trg = model(trg_img)
            
            # Gestione robusta: se è una tupla prendi il primo elemento, altrimenti tieni il tensore
            src_feats = out_src[0] if isinstance(out_src, (tuple, list)) else out_src
            trg_feats = out_trg[0] if isinstance(out_trg, (tuple, list)) else out_trg
            
            
            # --- Calcolo Loss ---
            # Recuperiamo dimensioni originali immagini per la normalizzazione interna alla loss
            src_h, src_w = src_img.shape[-2:]
            trg_h, trg_w = trg_img.shape[-2:]

            loss = criterion(
                src_featmaps=src_feats,
                trg_featmaps=trg_feats,
                src_kps=src_kps,           # Passiamo i tensori completi (B, N, 2)
                trg_kps=trg_kps,
                src_imgsize=(src_h, src_w), # Dimensioni reali immagine
                trg_imgsize=(trg_h, trg_w),
                npts=npts,                  # Numero punti validi per item
                softmax_temp=0.1,           # Temperatura
                enable_l2_norm=True
            )
            
            # Check NaN
            if torch.isnan(loss):
                print("Loss is NaN!")
                continue

            
            
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





