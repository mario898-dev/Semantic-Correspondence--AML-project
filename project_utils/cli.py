import argparse

def parse_eval_args():
    parser = argparse.ArgumentParser("AML Semantic Correspondence Evaluation")

    parser.add_argument(
        "--backbone",
        type=str,
        required=True,
        choices=["dinov2_vitb14", "dinov2_vitl14", "dinov3_vits16","dinov3_vitb16", "dinov3_vitl16", "sam_vitb"],
    )

    parser.add_argument(
        "--category",
        type=str,
        default="all",
        help="Categoria SPair-71k (es. cat, dog, car, all)"
    )

    parser.add_argument(
        "--max_pairs",
        type=int,
        default=None,
        help="Numero massimo di coppie (debug)"
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

    return parser.parse_args()
