import os
import argparse
import numpy as np
import pandas as pd


if __name__ == "__main__":

    # Parser creation
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_launch", type=int, help='')
    parser.add_argument("--shots_launch", type=int, help='')
    parser.add_argument("--lr_launch", type=float, help='')
    parser.add_argument("--model_launch", type=str, help='')
    parser.add_argument("--textual_launch", type=str, help='')
    parser.add_argument("--dataset_launch", type=str, help='')
    args = parser.parse_args()

    # Variables
    seed = args.seed_launch
    shot = args.shots_launch
    lr = args.lr_launch
    n_iters = 50
    position = "all"
    encoder = "vision"
    params = "q v"
    r = 2
    model_name = args.model_launch
    dataset = args.dataset_launch
    level = "class_name"
    backbone = None
    textual = args.textual_launch

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

    if model_name in ['clip', 'quilt', 'biomedclip', "vit_google"]:
        backbone = "ViT-B/16"
    elif model_name in ['uni']:
        backbone = "ViT-L/14"

    print(f"Run started: model {model_name}, lr {lr}, r {r}, seed {seed}")

    os.system(
        f"python3 main.py --root_path {root_path} \
        --dataset {dataset} --seed {seed} --shots {shot} --lr {lr} \
            --n_iters {n_iters} --position {position} --encoder {encoder}\
                --params {params} --r {r} --model_name {model_name} --num_classes {num_classes} --level {level} --backbone {backbone} --textual {textual}"
    )
