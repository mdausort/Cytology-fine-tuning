import os
import clip
import timm
import torch
import argparse
import open_clip
import numpy as np
import pandas as pd
from datasets import build_dataset
from transformers import AutoModelForImageClassification, AutoImageProcessor


if __name__ == "__main__":

    # Parser creation
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_launch", type=int, help="")
    parser.add_argument("--shots_launch", type=int, help="")
    parser.add_argument("--model_launch", type=str, help="")
    parser.add_argument("--dataset_launch", type=str, help="")
    parser.add_argument("--pourcent_launch", type=float, help="")
    parser.add_argument("--level_launch", type=str, help="")
    args = parser.parse_args()

    # Variables
    seed = args.seed_launch
    shot = args.shots_launch
    model = args.model_launch
    dataset = args.dataset_launch
    level = args.level_launch
    backbone = None
    pourcent = args.pourcent_launch

    if dataset == "kaggle1":
        num_classes = 4
        root_path = "/gpfs/projects/acad/danitim/elkhoury/manon/"

    elif dataset == "kaggle2":
        num_classes = 2
        root_path = "/gpfs/projects/acad/danitim/elkhoury/manon/"

    elif dataset == "sipakmed":
        num_classes = 5
        root_path = "/gpfs/projects/acad/danitim/elkhoury/manon/"

    elif dataset == "hicervix":
        df = pd.read_csv("/gpfs/projects/acad/danitim/elkhoury/manon/train.csv")
        class_list = np.unique(df.loc[:, level].tolist())
        num_classes = len(class_list)
        root_path = "/gpfs/projects/acad/danitim/elkhoury/manon/"
    else:
        raise RuntimeError("Wrong dataset")

    if model in ["clip", "quilt", "biomedclip", "vit_google"]:
        backbone = "ViT-B/16"
    elif model in ["uni"]:
        backbone = "ViT-L/14"

    # Model
    _, preprocess = clip.load(backbone)
    tokenizer = None

    if model == "clip":
        model_clip, preprocess = clip.load(backbone)
    elif model == "quilt":
        model_clip, preprocess, _ = open_clip.create_model_and_transforms(
            "hf-hub:wisdomik/QuiltNet-B-32"
        )
    elif model == "uni":
        model_clip = timm.create_model(
            "hf-hub:MahmoodLab/uni",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
        )
    elif model == "vit_google":
        _ = AutoImageProcessor.from_pretrained(
            "google/vit-base-patch16-224"
        )
        model_clip = AutoModelForImageClassification.from_pretrained(
            "google/vit-base-patch16-224"
        )
    elif model == "biomedclip":
        model_clip, preprocess, _ = open_clip.create_model_and_transforms(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        tokenizer = open_clip.get_tokenizer(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
    else:
        raise RuntimeError(
            "Wrong model name used. Try clip, conch, uni, biomedclip, vit_google or quilt."
        )

    level_name = level.replace("_", "")

    if pourcent > 0:
        if not os.path.exists(
            f"./{dataset}_{level_name}_{shot}_{pourcent}_pourcent.pt"
        ):
            dataset_all = build_dataset(
                dataset, root_path, shot, level, pourcent, preprocess
            )
            torch.save(
                dataset_all,
                f"./{dataset}_{level_name}_{shot}_{pourcent}_pourcent.pt",
            )
    else:
        if not os.path.exists(f"./{dataset}_{shot}_{shot}.pt"):
            dataset_all = build_dataset(
                dataset, root_path, shot, level, pourcent, preprocess
            )
            torch.save(dataset_all, f"./{dataset}_{shot}_{shot}.pt")
