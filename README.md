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


- Scarica SPair-71k nella cartella data
```bash
mkdir -p external/SD4Match/asset
cd external/SD4Match/asset
wget http://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz
tar -xvf SPair-71k.tar.gz
cd ../../../
```

- Per Evaluation:
```bash
!python eval.py --backbone [name] --category [name] --wandb
```
L'argomento backbone è obbligatorio\
L'argomento category se non specificato indica "all" categories\
L'argomento Wandb va inserito solo se si vuole utilizzare wandb

- Per training
```bash
!python train.py \
  --backbone (backbone name. Es. dinov2_vitb14) \
  --category ('all' or 'specific category es cat') \
  --trainable_layers (number) \
  --epochs (number) \
  --batch_size (dim 1) \
  --lr 1e-4 \
  --output_dir checkpoints \
  --wandb \
  --wandb_mode online
  --wandb_artifacts
```

- Per resume da un training interrotto
```bash
python train.py \
  --resume checkpoints/TRAIN-dinov2_vitb14-all-L1/last.pth \
  --backbone dinov2_vitb14 \
  --category all \
  --trainable_layers 1 \
  --epochs 10 \
  --batch_size 1 \
  --lr 1e-4 \
  --output_dir checkpoints \
  --wandb \
  --wandb_mode online \
  --wandb_artifacts
```

- IMG_SIZE
  - DINOV2: 518
  - DINOV3: 592
  - SAM: 1024
 
