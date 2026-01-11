import os
"""
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
"""

class Config:
    class DATASET:
        NAME = 'spair'
        ROOT = os.path.join(os.getcwd(), 'dataset')
        
        # Mapping Modello -> Risoluzione ottimale
        # Usa chiavi parziali per intercettare varianti (es. 'dinov2_vitb14' matcha 'dinov2')
        MODEL_RESOLUTIONS = {
            'dinov2': 518,
            'dinov3': 592,
            'sam': 592,
        }
        
        # Valore di default (fallback)
        IMG_SIZE = 518  
        
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]

        @classmethod
        def set_resolution(cls, model_name):
            """
            Imposta automaticamente IMG_SIZE cercando il nome del modello nel mapping.
            """
            model_name = model_name.lower()
            found = False
            for key, size in cls.MODEL_RESOLUTIONS.items():
                if key in model_name:
                    cls.IMG_SIZE = size
                    print(f" Config: Risoluzione impostata a {size}px per {model_name}")
                    found = True
                    break
            
            if not found:
                print(f"⚠️ Config: Modello '{model_name}' non riconosciuto nel mapping. Uso default: {cls.IMG_SIZE}px")

    class EVALUATOR:
        ALPHA = [0.05, 0.10, 0.15, 0.20]
        BY = 'bbox'