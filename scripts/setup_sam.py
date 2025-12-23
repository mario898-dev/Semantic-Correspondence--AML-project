# scripts/setup_sam.py
import os
import urllib.request

CKPT_DIR = "checkpoints"
CKPT_NAME = "sam_vit_b.pth"
CKPT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

os.makedirs(CKPT_DIR, exist_ok=True)
ckpt_path = os.path.join(CKPT_DIR, CKPT_NAME)

if not os.path.exists(ckpt_path):
    print("⬇️ Downloading SAM checkpoint...")
    urllib.request.urlretrieve(CKPT_URL, ckpt_path)
    print("SAM checkpoint downloaded")
else:
    print("SAM checkpoint already exists")
