# Semantic - Correspondence Project


## Come eseguire l'evaluation da Colab

- Montare google drive
```bash
from google.colab import drive
drive.mount("/content/drive")
```

- Vai in `/content` e clona la repo con i submodules:
```bash
%cd /content
git clone --recurse-submodules https://github.com/mario898-dev/Semantic-Correspondence--AML-project.git
cd Semantic-Correspondence--AML-project
```

- Installa le dipendenze
```bash
pip install -r requirements.txt
```


- Scarica SPair-71k nella cartella che usa SD4Match
```bash
mkdir -p external/SD4Match/asset
cd external/SD4Match/asset
wget http://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz
tar -xvf SPair-71k.tar.gz
cd ../../../
```

- Carica i pesi dal drive
```bash
SAM
!mkdir -p checkpoints/SAM
!cp /content/drive/MyDrive/AMLProject-data/weights_models/sam_vit_*.pth checkpoints/SAM/ 2>/dev/null ||true

DINOv3 (tutti i .pth che iniziano con dinov3_)
!mkdir -p checkpoints/DINOv3
!cp /content/drive/MyDrive/AMLProject-data/weights_models/dinov3_*.pth checkpoints/DINOv3/ 2>/dev/null || true

```

- Lancia l’evaluation:
```bash
!python eval.py --backbone [name] --category [name] --wandb
```
L'argomento backbone è obbligatorio\
L'argomento category se non specificato indica "all" categories\
L'argomento Wandb va inserito solo se si vuole utilizzare wandb

- Per training
```bash
!python train.py \
  --backbone [BACKBONE_NAME] \
  --category [CATEGORY|all] \
  --trainable_layers [NUM_LAYERS] \
  --batch_size [BATCH_SIZE] \
  --epochs [NUM_EPOCHS] \
  --lr [LEARNING_RATE] \
  --sigma [SIGMA] \
  --output_dir [OUTPUT_DIR] \
  [--wandb] \
  [--wandb_mode online|offline|disabled]

```

- IMG_SIZE
  - DINOV2: 518
  - DINOV3: 512
  - SAM: 1024
 
