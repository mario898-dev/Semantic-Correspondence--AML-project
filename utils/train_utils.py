import os
import random
import subprocess
import shutil
import numpy as np
import torch

# --- Setup Path ---
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# -------------------------
# Checkpoint utilities
# -------------------------
def _get_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": None,
    }
    if torch.cuda.is_available() and hasattr(torch.cuda, "get_rng_state_all"):
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state):
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and state.get("cuda") is not None and hasattr(torch.cuda, "set_rng_state_all"):
            torch.cuda.set_rng_state_all(state["cuda"])
    except Exception as e:
        print(f" Could not fully restore RNG state: {e}")


def _optimizer_to(optimizer, device):
    # Useful when loading optimizer state from CPU -> GPU
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def save_checkpoint(path, model, optimizer, epoch, global_step, train_loss, val_pck, best_pck, wandb_run_id, args_dict):
    """
    Save a complete checkpoint including Loss and PCK.
    Atomic save: writes to .tmp then calls os.replace.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        
        "train_loss": train_loss,
        "val_pck": val_pck,
        "best_pck": best_pck,
        
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
    """
    Load checkpoint and return useful info (including loss and pck).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device)

    missing, unexpected = model.load_state_dict(ckpt["model"], strict=strict)
    optimizer.load_state_dict(ckpt["optimizer"])
    _optimizer_to(optimizer, device)

    if load_rng and "rng" in ckpt:
        _set_rng_state(ckpt["rng"])

    info = {
        "epoch": ckpt.get("epoch", 0),
        "global_step": ckpt.get("global_step", 0),
        
        # Retrieve saved values (with safe defaults for backward compatibility)
        "train_loss": ckpt.get("train_loss", float("inf")),
        "val_pck": ckpt.get("val_pck", 0.0),
        "best_pck": ckpt.get("best_pck", 0.0), 
        # Fallback if best_pck doesn't exist but best_loss does (old checkpoints)
        "best_loss": ckpt.get("best_loss", float("inf")), 
        
        "wandb_run_id": ckpt.get("wandb_run_id", None),
        "args": ckpt.get("args", {}),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }
    return info


def sync_checkpoints_to_drive(run_dir: str, drive_run_dir: str):
    """
    Copy .pth files to Drive.
    """
    if not drive_run_dir:
        return

    os.makedirs(drive_run_dir, exist_ok=True)
    
    # Sync all .pth files in the directory
    files = [f for f in os.listdir(run_dir) if f.endswith(".pth")]

    for f in files:
        src = os.path.join(run_dir, f)
        dst = os.path.join(drive_run_dir, f)
        
        # Copy only if source is newer or destination is missing
        if not os.path.exists(dst) or os.path.getmtime(src) > os.path.getmtime(dst):
            try:
                # Try rsync if available (faster)
                subprocess.run(
                    ["rsync", "-a", "--partial", "--inplace", src, drive_run_dir + "/"],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
            except Exception:
                # Fallback: simple copy
                try:
                    tmp = dst + ".tmp"
                    shutil.copy2(src, tmp)
                    os.replace(tmp, dst)
                except Exception as e:
                    print(f"Error syncing to Drive ({f}): {e}")
