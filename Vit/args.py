import argparse
from src.config import ExpMode, ParMode

def parse_args():
    parser = argparse.ArgumentParser(description="Experiment Configuration")

    ################ experiment options ################

    parser.add_argument(
        "--exp-mode",
        "-e",
        type=str,
        choices=[e.value for e in ExpMode],
        default=ExpMode.TRAINING.value,
        help="Which experiment pipeline to run",
    )

    ################ common options ################
    parser.add_argument(
        "--par-mode",
        type=str,
        choices=[e.value for e in ParMode],
        default=ParMode.NONE.value,
        help="Parallelization mode",
    )
    
    parser.add_argument(
        "--reversible",
        action="store_true",
        default=False,
        help="Use reversible residual blocks",
    )

    parser.add_argument(
        "--inverse",
        action="store_true",
        default=False,
        help="Use inverse backward",
    )
    
    parser.add_argument(
        "--is_pareprop",
        action="store_true",
        default=False,
        help="Use Pareprop backward",
    )
    
    parser.add_argument(
        "--checkpointing",
        action="store_true",
        default=False,
        help="Use checkpointing to save memory",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar-10", "cifar-100", "cinic-10", "ImageNet-1K"],
        default="cifar-10",
        help="Dataset to use for training",
        
    )
################ model options ################
    parser.add_argument(
        "--num-hidden-layers",
        type=int,
        default=12,
        help="Number of hidden layers in the Transformer encoder",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=768,
        help="Dimensionality of the encoder layers and the pooler layer",
    )
    parser.add_argument(
        "--num-attention-heads",
        type=int,
        default=12,
        help="Number of attention heads for each attention layer in the Transformer encoder",
    )
# --- 変更点: BERT用引数を削除し、ViT用引数を追加 ---
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Input image size (squared)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=16,
        help="Patch size for ViT",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="Number of classification classes",
    )

    ################ training options ################
    parser.add_argument(
        "--autocast-dtype",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
        help="fp32, fp16, or bf16",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        metavar="N",
        help="Global batch size on each step (default: 256)",
    )
    parser.add_argument(
        "--microbatch-size",
        type=int,
        default=64,
        metavar="N",
        help="Micro-batch size for pipeline parallelism (default: 64)",
    )
    parser.add_argument(
        "--num-microbatches",
        type=int,
        default=4,
        metavar="N",
        help="Number of micro-batches for pipeline parallelism (default: 4)",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="Learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        metavar="M",
        help="Learning rate decay factor (default: 0.9)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        metavar="N",
        help="Number of epochs to train (default: 1)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, metavar="S", help="Random seed (default: 0)"
    )
    parser.add_argument(
        "--load-checkpoint",
        action="store_true",
        default=False,
        help="Load model from checkpoint",
    )
    
    parser.add_argument(
        "--save-checkpoint",
        action="store_true",
        default=False,
        help="Save model checkpoint",
    )

    ################ debug options ################
    parser.add_argument(
        "--debug-subset",
        action="store_true",
        default=False,
        help="[Debug] Use subset of dataset",
    )

    args = parser.parse_args()
    return args