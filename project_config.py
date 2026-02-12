import os


class Config:
    class DATASET:
        NAME = 'spair'
        ROOT = os.path.join(os.getcwd(), 'dataset')
        
        # Model -> optimal resolution mapping
        # Uses partial keys to match variants (e.g. 'dinov2_vitb14' matches 'dinov2')
        MODEL_RESOLUTIONS = {
            'dinov2': 518,
            'dinov3': 592,
            'sam': 592,
        }
        
        # Default value (fallback)
        IMG_SIZE = 518  
        
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]

        
        @classmethod
        def set_resolution(cls, model_name):
            """
            Automatically sets IMG_SIZE based on the model name in the mapping.
            """
            model_name = model_name.lower()
            found = False
            for key, size in cls.MODEL_RESOLUTIONS.items():
                if key in model_name:
                    cls.IMG_SIZE = size
                    #print(f" Config: Resolution set to {size}px for {model_name}")
                    found = True
                    break
            
            if not found:
                print(f"Config: Model '{model_name}' not recognized in mapping. Using default: {cls.IMG_SIZE}px")

        @classmethod
        def set_dataset(cls, dataset_name):
            """
            Automatically sets NAME.
            """

            dataset_name = dataset_name.lower()

            cls.NAME = dataset_name

    
    class EVALUATOR:
        ALPHA = [0.05, 0.10, 0.15, 0.20]
        BY = 'bbox'
