import os
import sys
import random
import subprocess
import shutil
import numpy as np
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


# -------------------------
# Checkpoint utilities
# -------------------------
def _get_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _set_rng_state(state):
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and state.get("cuda") is not None:
            torch.cuda.set_rng_state_all(state["cuda"])
    except Exception as e:
        print(f" Impossibile ripristinare RNG state in modo completo: {e}")


def _optimizer_to(optimizer, device):
    # utile quando carichi optimizer state da CPU -> GPU
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def save_checkpoint(path, model, optimizer, epoch, global_step, best_loss, wandb_run_id, args_dict):
    """
    Salvataggio atomico: scrive su .tmp e poi fa os.replace.
    Riduce il rischio di file corrotti/parziali (utile anche per copy su Drive).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        "best_loss": best_loss,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": _get_rng_state(),
        "wandb_run_id": wandb_run_id,
        "args": args_dict,
    }
    tmp_path = path + ".tmp"
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(path, model, optimizer, device, strict=True, load_rng=True):
    ckpt = torch.load(path, map_location=device)

    missing, unexpected = model.load_state_dict(ckpt["model"], strict=strict)
    optimizer.load_state_dict(ckpt["optimizer"])
    _optimizer_to(optimizer, device)

    if load_rng and "rng" in ckpt:
        _set_rng_state(ckpt["rng"])

    info = {
        "epoch": ckpt.get("epoch", 0),
        "global_step": ckpt.get("global_step", 0),
        "best_loss": ckpt.get("best_loss", float("inf")),
        "wandb_run_id": ckpt.get("wandb_run_id", None),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }
    return info


# -------------------------
# Drive sync utilities
# -------------------------
def sync_checkpoints_to_drive(run_dir: str, drive_run_dir: str):
    """
    Copia solo last.pth e best.pth su Drive a fine epoca.
    - Usa rsync se disponibile (incrementale).
    - Fallback con shutil (copia + replace atomico lato destinazione).
    """
    if not drive_run_dir:
        return

    os.makedirs(drive_run_dir, exist_ok=True)
    files = ["last.pth", "best.pth"]

    # prova rsync (migliore)
    try:
        for f in files:
            src = os.path.join(run_dir, f)
            if os.path.isfile(src):
                subprocess.run(
                    ["rsync", "-a", "--partial", "--inplace", src, drive_run_dir + "/"],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        return
    except Exception:
        pass

    # fallback: copia semplice (più lenta ma affidabile)
    for f in files:
        src = os.path.join(run_dir, f)
        if os.path.isfile(src):
            dst = os.path.join(drive_run_dir, f)
            tmp = dst + ".tmp"
            shutil.copy2(src, tmp)
            os.replace(tmp, dst)


def run_training(args):
    # 1. Setup Dispositivo e Modello
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device training: {device}")

    print(f"Costruzione modello {args.backbone} con {args.trainable_layers} layer addestrabili...")
    model = build_model(args.backbone, device, num_trainable_layers=args.trainable_layers)
    model.train()

    # 2. Dataset e DataLoader
    print(f"Caricamento SPair-71k (split: train, category: {args.category})...")
    train_dataset = SPairDataset(cfg=Config, split="trn", category=args.category)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=None
    )
    print(f"Training su {len(train_dataset)} coppie.")

    # 3. Optimizer e Loss
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params_to_optimize, lr=args.lr)
    criterion = FeatMapLoss()
    print(f"Parametri da ottimizzare: {len(params_to_optimize)} tensor(s).")

    # 4. Output directory (run folder)
    os.makedirs(args.output_dir, exist_ok=True)
    run_name = f"TRAIN-{args.backbone}-{args.category}-L{args.trainable_layers}"
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    last_ckpt_path = os.path.join(run_dir, "last.pth")
    best_ckpt_path = os.path.join(run_dir, "best.pth")

    # 4b. Drive sync (opzionale): setta env var DRIVE_SYNC_DIR per abilitarlo
    # es: DRIVE_SYNC_DIR=/content/drive/MyDrive/AMLProject-data/checkpoints
    drive_root = os.environ.get("DRIVE_SYNC_DIR", "").strip()
    drive_run_dir = os.path.join(drive_root, run_name) if drive_root else None
    if drive_run_dir:
        os.makedirs(drive_run_dir, exist_ok=True)
        print(f"✅ Drive sync attivo: {drive_run_dir}")
    else:
        print("ℹ️ Drive sync non attivo (setta env DRIVE_SYNC_DIR per abilitarlo).")

    # 5. (Resume) prima di init wandb (per recuperare run_id)
    start_epoch = 0
    global_step = 0
    best_loss = float("inf")
    wandb_run_id = getattr(args, "wandb_run_id", None)  # opzionale

    if getattr(args, "resume", None):
        if os.path.isfile(args.resume):
            print(f" Resume da checkpoint: {args.resume}")
            info = load_checkpoint(
                args.resume,
                model=model,
                optimizer=optimizer,
                device=device,
                strict=True,
                load_rng=True
            )
            start_epoch = info["epoch"] + 1
            global_step = info["global_step"]
            best_loss = info["best_loss"]
            if info["wandb_run_id"] is not None:
                wandb_run_id = info["wandb_run_id"]

            if len(info["missing_keys"]) or len(info["unexpected_keys"]):
                print(" Chiavi model state non perfettamente allineate:")
                print("  missing:", info["missing_keys"])
                print("  unexpected:", info["unexpected_keys"])

            print(f" Ripartenza: start_epoch={start_epoch}, global_step={global_step}, best_loss={best_loss:.6f}")
        else:
            print(f" args.resume passato ma file non trovato: {args.resume}")

    # 6. WandB Init (con resume)
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

        wandb.define_metric("global_step")
        wandb.define_metric("train_loss", step_metric="global_step")
        wandb.define_metric("epoch_avg_loss", step_metric="global_step")

    # 7. Training Loop
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

            valid_mask = (
                (src_kps[:, 0] >= 0) & (src_kps[:, 1] >= 0) &
                (trg_kps[:, 0] >= 0) & (trg_kps[:, 1] >= 0)
            )

            npts = valid_mask.sum(dim=1)
            if npts.sum() == 0:
                continue

            optimizer.zero_grad(set_to_none=True)

            out_src = model(src_img)
            out_trg = model(trg_img)

            src_feats = out_src[0] if isinstance(out_src, (tuple, list)) else out_src
            trg_feats = out_trg[0] if isinstance(out_trg, (tuple, list)) else out_trg

            src_h, src_w = src_img.shape[-2:]
            trg_h, trg_w = trg_img.shape[-2:]

            loss = criterion(
                src_featmaps=src_feats,
                trg_featmaps=trg_feats,
                src_kps=src_kps,
                trg_kps=trg_kps,
                src_imgsize=(src_h, src_w),
                trg_imgsize=(trg_h, trg_w),
                npts=npts,
                softmax_temp=0.1,
                enable_l2_norm=True
            )

            if torch.isnan(loss):
                print("Loss is NaN!")
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

        avg_loss = (epoch_loss / num_batches) if num_batches > 0 else 0.0
        print(f"Fine Epoca {epoch+1} - Avg Loss: {avg_loss:.6f}")

        if args.wandb:
            wandb.log({"epoch_avg_loss": avg_loss, "global_step": global_step}, step=global_step)

        # 8. Save LAST checkpoint (resume)
        save_checkpoint(
            path=last_ckpt_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            best_loss=best_loss,
            wandb_run_id=wandb_run_id,
            args_dict=vars(args),
        )
        print(f" Salvato LAST checkpoint: {last_ckpt_path}")

        # 9. Save BEST checkpoint (qui basato su loss; se poi vuoi PCK, lo cambiamo)
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                path=best_ckpt_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                best_loss=best_loss,
                wandb_run_id=wandb_run_id,
                args_dict=vars(args),
            )
            print(f" Nuovo BEST checkpoint (loss={best_loss:.6f}): {best_ckpt_path}")

        # 9b. Sync su Drive a fine epoca (se attivo)
        if drive_run_dir:
            sync_checkpoints_to_drive(run_dir, drive_run_dir)
            print(f" ✅ Sync su Drive completata (fine epoca): {drive_run_dir}")

        # 10. (Opzionale) log checkpoint come Artifact su wandb
        if args.wandb and getattr(args, "wandb_artifacts", False):
            art_last = wandb.Artifact("ckpt-last", type="checkpoint")
            art_last.add_file(last_ckpt_path)
            wandb.log_artifact(art_last)

            if os.path.isfile(best_ckpt_path):
                art_best = wandb.Artifact("ckpt-best", type="checkpoint")
                art_best.add_file(best_ckpt_path)
                wandb.log_artifact(art_best)

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    args = parse_train_args()
    run_training(args)
