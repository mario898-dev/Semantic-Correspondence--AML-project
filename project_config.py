import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

class Config:
    class DATASET:
        NAME = 'spair'
        # Percorso dove viene scaricato SPair-71k 
        ROOT = os.path.join(os.getcwd(), 'dataset')
        
        IMG_SIZE = 592
        
        MEAN = False
        NORM_MEAN = [0.485, 0.456, 0.406]
        NORM_STD = [0.229, 0.224, 0.225]
        STD = [0.229, 0.224, 0.225]

    class EVALUATOR:
        ALPHA = [0.05, 0.10, 0.15, 0.20]
        BY = 'bbox'
