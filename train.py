import os
import shutil
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
from utils.loss import FeatMapLoss
from utils.train_utils import save_checkpoint, load_checkpoint, sync_checkpoints_to_drive
from utils.validation import validate_epoch


def run_training(args):

    Config.DATASET.set_resolution(args.backbone)
    
    # --- 1. DEVICE AND MODEL SETUP ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device training: {device}")

    print(f"Building model {args.backbone} with {args.trainable_layers} trainable layers...")
    model = build_model(args.backbone, device, num_trainable_layers=args.trainable_layers, weights=args.weights)
    model.train()

    # --- 2. DATASET AND DATALOADER ---
    print(f"Loading SPair-71k (split: train, category: {args.category})...")
    train_dataset = SPairDataset(cfg=Config, split="trn", category=args.category)
    
    # Optimized configuration for A100 (more workers, shuffle enabled)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=None
    )
    print(f"Training on {len(train_dataset)} pairs.")

    print(f"IMG_SIZE used in train: {Config.DATASET.IMG_SIZE}")
    # --- 3. OPTIMIZER E LOSS ---
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params_to_optimize, lr=args.lr)
    criterion = FeatMapLoss()
    print(f"Parameters to optimize: {len(params_to_optimize)} tensor(s).")

    # --- 4. OUTPUT DIR E VARIABILI STATO ---
    os.makedirs(args.output_dir, exist_ok=True)
    run_name = f"TRAIN-{args.backbone}-{args.category}-L{args.trainable_layers}"
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Tracking variables
    best_pck = 0.0
    training_history = []

    # Drive Sync setup (optional)
    drive_root = os.environ.get("DRIVE_SYNC_DIR", "").strip()
    drive_run_dir = os.path.join(drive_root, run_name) if drive_root else None
    if drive_run_dir:
        os.makedirs(drive_run_dir, exist_ok=True)
        print(f"Drive sync enabled: {drive_run_dir}")

    # --- 5. RESUME (Optional) ---
    start_epoch = 0
    global_step = 0
    wandb_run_id = getattr(args, "wandb_run_id", None)

    if getattr(args, "resume", None):
        if os.path.isfile(args.resume):
            print(f" Resuming from checkpoint: {args.resume}")
            info = load_checkpoint(args.resume, model, optimizer, device)
            start_epoch = info["epoch"] + 1
            global_step = info["global_step"]
            
            # Recover Best PCK (backward compatibility handling)
            best_pck = info.get("best_pck", 0.0)
            if best_pck == 0.0 and "best_loss" in info:
                # If checkpoint used best_loss as geometric metric
                best_pck = info["best_loss"]
            
            # Recover history
            if "args" in info and isinstance(info["args"], dict):
                training_history = info["args"].get("history", [])
                # If history exists, recalculate the real best_pck from it for safety
                if training_history:
                    hist_best = max([h['val_pck'] for h in training_history])
                    if hist_best > best_pck:
                        best_pck = hist_best

            wandb_run_id = info.get("wandb_run_id", wandb_run_id)
            print(f" Resuming: Epoch {start_epoch}, Previous Best PCK: {best_pck:.2f}%")
        else:
            print(f" args.resume specified but file not found: {args.resume}")

    # --- 6. WANDB INIT ---
    if args.wandb:
        if wandb_run_id is None:
            wandb_run_id = wandb.util.generate_id()

        wandb.init(
            project="AML-Semantic-Correspondence",
            id=wandb_run_id,
            resume="allow",
            name=run_name,
            mode=args.wandb_mode,
            config=vars(args),
        )

    # --- 7. TRAINING LOOP ---
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*20} Epoch {epoch+1}/{args.epochs} {'='*20}")
        epoch_loss = 0.0
        num_batches = 0

        model.train()
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}")

        for batch in pbar:
            src_img = batch["src_img"].to(device)
            trg_img = batch["trg_img"].to(device)
            src_kps = batch["src_kps"].to(device)
            trg_kps = batch["trg_kps"].to(device)

            
            npts = batch['n_pts']
            if npts.sum() == 0:
                continue

            optimizer.zero_grad(set_to_none=True)

            out_src = model(src_img)
            out_trg = model(trg_img)

            src_feats = out_src[0] if isinstance(out_src, (tuple, list)) else out_src
            trg_feats = out_trg[0] if isinstance(out_trg, (tuple, list)) else out_trg

            # Compute Loss
            loss = criterion(
                src_featmaps=src_feats,
                trg_featmaps=trg_feats,
                src_kps=src_kps,
                trg_kps=trg_kps,
                src_imgsize=src_img.shape[-2:],
                trg_imgsize=trg_img.shape[-2:],
                npts=npts,
                softmax_temp=0.04
            )

            if torch.isnan(loss):
                print("Warning: NaN loss detected, skipping batch.")
                continue

            loss.backward()
            optimizer.step()

            loss_val = float(loss.item())
            epoch_loss += loss_val
            num_batches += 1
            global_step += 1

            pbar.set_postfix({"loss": f"{loss_val:.4f}"})

            if args.wandb:
                wandb.log({"train_loss": loss_val, "global_step": global_step}, step=global_step)

        # Compute Average Loss
        avg_loss = (epoch_loss / num_batches) if num_batches > 0 else 0.0
        print(f"End of Epoch {epoch+1} - Avg Loss: {avg_loss:.6f}")

        # --- 8. FULL VALIDATION ---
        # Run validation on the full set to get the real PCK
        current_pck = validate_epoch(model, device, args.category)

        # Update history
        epoch_stats = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_pck": current_pck
        }
        training_history.append(epoch_stats)

        if args.wandb:
            wandb.log({
                "epoch_avg_loss": avg_loss, 
                "val_pck": current_pck, 
                "epoch": epoch
            }, step=global_step)

        # --- 9. SAVE CHECKPOINT (One per epoch) ---
        epoch_ckpt_name = f"checkpoint_ep{epoch}.pth"
        epoch_ckpt_path = os.path.join(run_dir, epoch_ckpt_name)

        # Save everything: state, optimizer, Loss, PCK, full history
        save_checkpoint(
            path=epoch_ckpt_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            train_loss=avg_loss,
            val_pck=current_pck,
            best_pck=best_pck,
            wandb_run_id=wandb_run_id,
            args_dict={**vars(args), "history": training_history}
        )
        print(f"Checkpoint saved: {epoch_ckpt_name} (Loss: {avg_loss:.4f}, PCK: {current_pck:.2f}%)")

        # --- 10. BEST MODEL MANAGEMENT ---
        # If current PCK is the best ever, update best.pth
        if current_pck > best_pck:
            best_pck = current_pck
            best_ckpt_path = os.path.join(run_dir, "best.pth")
            
            # Physical file copy
            shutil.copyfile(epoch_ckpt_path, best_ckpt_path)
            print(f"NEW BEST MODEL (PCK: {best_pck:.2f}%) -> saved to best.pth")
            
            # Immediate Drive sync for the best model
            if drive_run_dir:
                sync_checkpoints_to_drive(run_dir, drive_run_dir)

    print("\nTraining Completed.")
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    args = parse_train_args()
    run_training(args)





