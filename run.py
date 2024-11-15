import os
import argparse
import numpy as np
import pandas as pd


if __name__ == "__main__":

    # Parser creation
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_launch", type=int, default=1, help="Seed number")
    parser.add_argument("--shots_launch", type=int, default=16, help="Shot number")
    parser.add_argument("--lr_launch", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--iterations", type=int, default=50, help="Number of iteration"
    )
    parser.add_argument(
        "--parameters",
        type=str,
        default="q v",
        help="Layers of the model on which LoRA will be applied",
    )
    parser.add_argument(
        "--rank_launch",
        type=int,
        default=2,
        help="Rank of matrices on which LoRA will be applied",
    )

    parser.add_argument(
        "--model_launch",
        type=str,
        default="clip",
        help="Name of the model used",
        choices=["clip", "quilt", "biomedclip", "vit_google", "uni"],
    )
    parser.add_argument(
        "--textual_launch",
        type=str,
        default="False",
        help="If True, the classifier is initialized with textual embeddings. If False, the textual information is ignored.",
    )
    parser.add_argument(
        "--dataset_launch",
        type=str,
        default="kaggle1",
        help="Name of the dataset used",
        choices=["kaggle1", "kaggle2", "sipakmed", "hicervix"],
    )
    parser.add_argument(
        "--level_launch",
        type=str,
        default="level_1",
        help="This is the level of the hierarchical tree to capture different fine-grained subtype information. Only applicable in the case of hicervix.",
    )
    parser.add_argument(
        "--percent_launch",
        type=float,
        default=0.0,
        help="Percentage of the dataset considered. Used for the third experiment.",
    )

    parser.add_argument(
        "--task_launch",
        type=str,
        default="lora",
        help="Task name",
        choices=["classifier", "lora", "percentage_lora"],
    )
    args = parser.parse_args()

    # Variables
    seed = args.seed_launch
    shot = args.shots_launch
    lr = args.lr_launch
    n_iters = args.iterations
    params = args.parameters
    position = "all"
    encoder = "vision"
    r = args.rank_launch

    model_name = args.model_launch
    textual = args.textual_launch
    dataset = args.dataset_launch
    level = args.level_launch
    percent = args.percent_launch

    task = args.task_launch

    if dataset == "kaggle1":
        num_classes = 4
        root_path = "/gpfs/projects/acad/danitim/elkhoury/manon/"  # TO CHANGE

    elif dataset == "kaggle2":
        num_classes = 2
        root_path = "/gpfs/projects/acad/danitim/elkhoury/manon/"  # TO CHANGE

    elif dataset == "sipakmed":
        num_classes = 5
        root_path = "/gpfs/projects/acad/danitim/elkhoury/manon/"  # TO CHANGE

    elif dataset == "hicervix":
        df = pd.read_csv(
            "/gpfs/projects/acad/danitim/elkhoury/manon/train.csv"
        )  # TO CHANGE

        if level == "level_3":
            class_list_2 = sorted(np.unique(df.loc[:, "level_2"].dropna().tolist()))
            class_list_3 = sorted(np.unique(df.loc[:, "level_3"].dropna().tolist()))

            combined_class_list = np.append(class_list_2, class_list_3)

            cleaned_class_list = pd.Series(combined_class_list).dropna()

            class_list = sorted(np.unique(cleaned_class_list).tolist())

        elif level == "level_2":
            class_list_2 = sorted(np.unique(df.loc[:, "level_2"].dropna().tolist()))
            class_list_1 = sorted(np.unique(df.loc[:, "level_1"].dropna().tolist()))

            combined_class_list = np.append(class_list_2, class_list_1)

            cleaned_class_list = pd.Series(combined_class_list).dropna()

            class_list = sorted(np.unique(cleaned_class_list).tolist())

        else:
            class_list = np.unique(df.loc[:, level].tolist())

        num_classes = len(class_list)
        root_path = "/gpfs/projects/acad/danitim/elkhoury/manon/"  # TO CHANGE

    else:
        raise RuntimeError("Wrong dataset")

    if model_name in ["clip", "quilt", "biomedclip", "vit_google"]:
        backbone = "ViT-B/16"
    elif model_name in ["uni"]:
        backbone = "ViT-L/14"

    if task == "percentage_lora":
        backbone = "ViT-L/14"

    print(f"Run started: model {model_name}, lr {lr}, r {r}, seed {seed}")

    os.system(
        f"python3 main.py --root_path {root_path} \
        --dataset {dataset} --seed {seed} --shots {shot} --lr {lr} \
            --n_iters {n_iters} --position {position} --encoder {encoder} --percentage {percent}\
                --params {params} --r {r} --model_name {model_name} --num_classes {num_classes}\
                    --level {level} --backbone {backbone} --textual {textual} --task {task}"
    )
