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
        help="SPair-71k category (e.g. cat, dog, car, all)"
    )
    

    parser.add_argument(
        "--weights",
        type=str,
        default=None
    )
  

    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable wandb logging"
    )

    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="Wandb mode"
    )

    parser.add_argument(
        "--extract_layer",
        type=int,
        default=None,
        help="Layer index to extract (SAM only). E.g.: 8 for layer 9."
    )

    parser.add_argument(
        "--use_window_soft",
        type=int,
        default=0,
        help='If set, use Window Soft Argmax instead of classic Argmax'
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="spair",
        help='Select which dataset to use'
    )

    parser.add_argument(
        "--split",
        type=str,
        default='test',
        help='Select which split to use: "test", "test_cross_species", "test_cross_family"'
    )
    
    return parser.parse_args()

def parse_train_args():
    parser = argparse.ArgumentParser("AML Semantic Correspondence Training")

    # --- Common arguments (Model and Data) ---
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
        help="SPair-71k category for training (default: all)"
    )

    # --- Training arguments ---
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for finetuned layers"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Total number of epochs"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (recommended 1 to handle variable numbers of keypoints)"
    )

    parser.add_argument(
        "--trainable_layers",
        type=int,
        default=1,
        help="Number of final backbone blocks to unfreeze for fine-tuning"
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
        help="Sigma of the Gaussian for the Window Soft Target Loss"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Directory to save model weights"
    )

    # --- WandB ---
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable WandB logging"
    )

    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="WandB mode"
    )

    parser.add_argument("--resume", type=str, default=None,
                    help="Path to a .pth checkpoint (e.g. .../last.pth) to resume training")
    
    parser.add_argument("--wandb_run_id", type=str, default=None,
                    help="(Optional) Force a wandb run id. If not given, it is saved/reused from checkpoint.")
    
    parser.add_argument("--wandb_artifacts", action="store_true",
                    help="If active, upload last/best checkpoint as wandb Artifacts (useful on Colab)")

    return parser.parse_args()
