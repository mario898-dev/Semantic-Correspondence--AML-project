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


- Scarica SPair-71k 
```bash
!mkdir -p dataset
!rm -rf dataset/SPair-71k
%cd dataset
!wget -nc http://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz
!tar -xzf SPair-71k.tar.gz
!rm -f SPair-71k.tar.gz
%cd ..
```

- Scarica PF-Willow 
```bash
%cd /content/Semantic-Correspondence--AML-project/dataset

!mkdir -p pf-willow
%cd pf-willow

!wget https://www.di.ens.fr/willow/research/proposalflow/dataset/PF-dataset.zip
!unzip -q PF-dataset.zip
!rm PF-dataset.zip

!wget https://www.robots.ox.ac.uk/~xinghui/sd4match/test_pairs.csv

%cd ../..
```
- Scarica PF-Pascal 
```bash
%cd /content/Semantic-Correspondence--AML-project/dataset

!mkdir -p pf-pascal
%cd pf-pascal

!wget -nc https://www.di.ens.fr/willow/research/proposalflow/dataset/PF-dataset-PASCAL.zip
!unzip -q -n PF-dataset-PASCAL.zip
!rm PF-dataset-PASCAL.zip

!wget -N https://www.robots.ox.ac.uk/~xinghui/sd4match/pf-pascal_image_pairs.zip
!unzip -q -o pf-pascal_image_pairs.zip
!rm pf-pascal_image_pairs.zip

!find . -name "*_pairs.csv" -exec mv {} . \;

%cd ../..
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
 
