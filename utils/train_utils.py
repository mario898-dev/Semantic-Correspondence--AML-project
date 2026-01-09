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
        print(f" Impossibile ripristinare RNG state in modo completo: {e}")


def _optimizer_to(optimizer, device):
    # utile quando carichi optimizer state da CPU -> GPU
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def save_checkpoint(path, model, optimizer, epoch, global_step, train_loss, val_pck, best_pck, wandb_run_id, args_dict):
    """
    Salva un checkpoint completo includendo Loss e PCK.
    Salvataggio atomico: scrive su .tmp e poi fa os.replace.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        
        # --- METADATA AGGIUNTIVI ---
        "train_loss": train_loss,  # Loss media dell'epoca
        "val_pck": val_pck,        # PCK calcolata in validazione
        "best_pck": best_pck,      # Miglior PCK vista finora
        # ---------------------------
        
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
    Carica il checkpoint e restituisce le info utili (inclusi loss e pck).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint non trovato: {path}")

    ckpt = torch.load(path, map_location=device)

    missing, unexpected = model.load_state_dict(ckpt["model"], strict=strict)
    optimizer.load_state_dict(ckpt["optimizer"])
    _optimizer_to(optimizer, device)

    if load_rng and "rng" in ckpt:
        _set_rng_state(ckpt["rng"])

    info = {
        "epoch": ckpt.get("epoch", 0),
        "global_step": ckpt.get("global_step", 0),
        
        # Recupero i valori salvati (con valori di default sicuri per retro-compatibilità)
        "train_loss": ckpt.get("train_loss", float("inf")),
        "val_pck": ckpt.get("val_pck", 0.0),
        "best_pck": ckpt.get("best_pck", 0.0), 
        # Fallback se best_pck non esiste ma c'è best_loss (vecchi checkpoint)
        "best_loss": ckpt.get("best_loss", float("inf")), 
        
        "wandb_run_id": ckpt.get("wandb_run_id", None),
        "args": ckpt.get("args", {}),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }
    return info


def sync_checkpoints_to_drive(run_dir: str, drive_run_dir: str):
    """
    Copia i file .pth su Drive.
    """
    if not drive_run_dir:
        return

    os.makedirs(drive_run_dir, exist_ok=True)
    
    # Sincronizza tutti i .pth presenti nella cartella
    files = [f for f in os.listdir(run_dir) if f.endswith(".pth")]

    for f in files:
        src = os.path.join(run_dir, f)
        dst = os.path.join(drive_run_dir, f)
        
        # Copia solo se sorgente più recente o destinazione assente
        if not os.path.exists(dst) or os.path.getmtime(src) > os.path.getmtime(dst):
            try:
                # Prova rsync se disponibile (più veloce)
                subprocess.run(
                    ["rsync", "-a", "--partial", "--inplace", src, drive_run_dir + "/"],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
            except Exception:
                # Fallback copia semplice
                try:
                    tmp = dst + ".tmp"
                    shutil.copy2(src, tmp)
                    os.replace(tmp, dst)
                except Exception as e:
                    print(f"Errore sync Drive ({f}): {e}")
