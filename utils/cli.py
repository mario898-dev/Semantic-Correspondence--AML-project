import argparse

def parse_eval_args():
    parser = argparse.ArgumentParser("AML Semantic Correspondence Evaluation")

    parser.add_argument(
        "--backbone",
        type=str,
        required=True,
        choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov3_vits16","dinov3_vitb16", "dinov3_vitl16", "sam_vitb"],
    )

    parser.add_argument(
        "--category",
        type=str,
        default="all",
        help="Categoria SPair-71k (es. cat, dog, car, all)"
    )
    

    parser.add_argument(
        "--weights",
        type=str,
        default=None
    )
  

    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Abilita wandb logging"
    )

    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="Modalità wandb"
    )

    parser.add_argument(
        "--extract_layer",
        type=int,
        default=None,
        help="Indice del layer da estrarre (solo per SAM). Es: 8 per il layer 9."
    )

    return parser.parse_args()

def parse_train_args():
    parser = argparse.ArgumentParser("AML Semantic Correspondence Training")

    # --- Argomenti Comuni (Modello e Dati) ---
    parser.add_argument(
        "--backbone",
        type=str,
        required=True,
        choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", 
                 "dinov3_vits16","dinov3_vitb16", "dinov3_vitl16", "sam_vitb"],
    )
    
    parser.add_argument(
        "--category",
        type=str,
        default="all",
        help="Categoria SPair-71k per il training (default: all)"
    )

    # --- Argomenti Training ---
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning Rate per i layer finetunati"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Numero totale di epoche"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (consigliato 1 per gestire numeri variabili di keypoints)"
    )

    parser.add_argument(
        "--trainable_layers",
        type=int,
        default=1,
        help="Numero di blocchi finali del backbone da sbloccare per il fine-tuning"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None
    )
    
    parser.add_argument(
        "--sigma",
        type=float,
        default=2.0,
        help="Sigma della Gaussiana per la Window Soft Target Loss"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Cartella dove salvare i pesi del modello"
    )

    # --- WandB ---
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Abilita il logging su WandB"
    )

    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="Modalità WandB"
    )

    parser.add_argument("--resume", type=str, default=None,
                    help="Path a un checkpoint .pth (es. .../last.pth) per riprendere il training")
    
    parser.add_argument("--wandb_run_id", type=str, default=None,
                    help="(Opzionale) Forza un run id wandb. Se non dato, viene salvato/riusato dal checkpoint.")
    
    parser.add_argument("--wandb_artifacts", action="store_true",
                    help="Se attivo, carica last/best checkpoint come wandb Artifacts (utile su Colab)")

    return parser.parse_args()
