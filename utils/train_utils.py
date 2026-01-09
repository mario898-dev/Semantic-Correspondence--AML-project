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
