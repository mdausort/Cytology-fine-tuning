import random
import argparse
import numpy as np
import torch


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1, help="Seed number")

    # Dataset arguments
    parser.add_argument(
        "--root_path",
        type=str,
        default="",
        help="Path of your root directory. We put our dataset in it.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="kaggle1",
        help="Name of the dataset used",
        choices=["kaggle1", "kaggle2", "sipakmed", "hicervix"],
    )
    parser.add_argument("--shots", type=int, default=16, help="Shot number")
    parser.add_argument(
        "--percentage",
        type=float,
        default=0.0,
        help="Percentage of the dataset considered. Used for the third experiment.",
    )
    parser.add_argument(
        "--textual",
        type=str,
        default="False",
        help="If True, the classifier is initialized with textual embeddings. If False, the textual information is ignored.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="lora",
        help="Task name",
        choices=["classifier", "lora", "percentage_lora"],
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="clip",
        help="Name of the model used",
        choices=["clip", "quilt", "biomedclip", "vit_google", "uni"],
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=2,
        help="Number of classes considered for the classification task",
    )
    parser.add_argument(
        "--level",
        type=str,
        default="level_1",
        help="This is the level of the hierarchical tree to capture different fine-grained subtype information. Only applicable in the case of hicervix.",
        choices=["level_1", "level_2", "level_3", "class_name"],
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="ViT-B/16",
        help="Configuration of the model's backbone",
        choices=["ViT-L/14", "ViT-B/16"],
    )

    # Training arguments
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--n_iters", type=int, default=500, help="Number of iterations")
    parser.add_argument("--batch_size", type=int, default=32, help="Size of the batch")

    # Argument definition
    parser.add_argument(
        "--position",
        type=str,
        default="all",
        help="where to put the LoRA modules",
        choices=["bottom", "mid", "up", "half-up", "half-bottom", "all", "top3"],
    )

    parser.add_argument(
        "--encoder",
        type=str,
        default="both",
        choices=["text", "vision", "both"],
        help="It is the part of the model on which we want apply LoRA, either on the visual or textual part.",
    )

    parser.add_argument(
        "--params",
        type=str,
        metavar="N",
        nargs="+",
        default=["q", "k", "v"],
        help="list of attention matrices where putting a LoRA",
    )

    parser.add_argument(
        "--r", type=int, default=2, help="the rank of the low-rank matrices"
    )

    parser.add_argument("--alpha", default=1, type=int, help="scaling (see LoRA paper)")

    parser.add_argument(
        "--dropout_rate",
        default=0.25,
        type=float,
        help="dropout rate applied before the LoRA module",
    )

    parser.add_argument(
        "--save_path",
        default=None,
        help="path to save the lora modules after training, not saved if None",
    )
    parser.add_argument(
        "--filename",
        default="lora_weights",
        help="file name to save the lora weights (.pt extension will be added)",
    )

    parser.add_argument(
        "--eval_only",
        default=False,
        action="store_true",
        help="only evaluate the LoRA modules (save_path should not be None)",
    )

    args = parser.parse_args()

    return args
