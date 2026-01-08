import os

class Config:
    class DATASET:
        NAME = 'spair'
        # Percorso dove viene scaricato SPair-71k 
        ROOT = os.path.join(os.getcwd(), 'dataset')
        
        IMG_SIZE = 592
        
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]

    class EVALUATOR:
        ALPHA = [0.05, 0.10, 0.15, 0.20]
        BY = 'bbox'
