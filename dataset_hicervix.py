import os
import torch
import argparse
import numpy as np
import pandas as pd
from datasets import build_dataset  # type: ignore


if __name__ == "__main__":

    # Parser creation
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_launch", type=int, help="Seed number")
    parser.add_argument("--shots_launch", type=int, help="Shot number")
    parser.add_argument(
        "--level_launch",
        default="level_1",
        type=str,
        help="This is the level of the hierarchical tree to capture different fine-grained subtype information. Only applicable in the case of hicervix.",
    )
    parser.add_argument(
        "--percent_launch",
        type=float,
        help="Percentage of the dataset considered. Used for the third experiment.",
    )

    args = parser.parse_args()

    # Variables
    seed = args.seed_launch
    shot = args.shots_launch
    dataset = "hicervix"
    level = args.level_launch
    percent = args.percent_launch

    if dataset == "hicervix":

        df = pd.read_csv(
            "path_of_dataset/train.csv"
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

        root_path = "path_of_dataset"  # TO CHANGE

    else:
        raise RuntimeError("Wrong dataset")

    level_name = level.replace("_", "")

    if percent > 0:
        if not os.path.exists(
            f"./{dataset}_{seed}_{shot}_{level_name}_{percent}_percent.pt"
        ):
            dataset_all = build_dataset(dataset, root_path, shot, level, percent)
            torch.save(
                dataset_all,
                f"./{dataset}_{seed}_{shot}_{level_name}_{percent}_percent.pt",
            )
    else:
        if not os.path.exists(f"./{dataset}_{seed}_{shot}_{level_name}.pt"):
            dataset_all = build_dataset(dataset, root_path, shot, level)
            torch.save(dataset_all, f"./{dataset}_{seed}_{shot}_{level_name}.pt")
