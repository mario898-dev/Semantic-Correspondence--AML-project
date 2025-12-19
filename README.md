# Semantic - Correspondence Project


## Come eseguire l'evaluation da Colab

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

- Lancia l’evaluation:
```bash
!python eval.py
```
