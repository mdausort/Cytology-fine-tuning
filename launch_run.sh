#!/bin/bash

lrs=("1e-2" "3e-3" "1e-3" "3e-4" "1e-4")
seeds=("1" "2" "3")
models=("clip" "uni" "vit_google" "biomedclip" "quilt")
shots=("0" "1" "2" "4" "8" "16" "32" "40" "50")

index=$SLURM_ARRAY_TASK_ID
total_models=${#models[@]}
total_shots=${#shots[@]}
total_seeds=${#seeds[@]}
total_lrs=${#lrs[@]}

model_idx=$(( (index / total_shots) % total_models ))
shot_idx=$(( index % total_shots ))
seed_idx=$(( (index / (total_models * total_shots)) % total_seeds ))
lr_idx=$(( index / (total_models * total_shots * total_seeds) ))

lr=${lrs[lr_idx]}
seed=${seeds[seed_idx]}
model=${models[model_idx]}
shot=${shots[shot_idx]}

module load devel/python/3.9.13  # TO CHANGE
source .env/bin/activate  # TO CHANGE

# Classifier line
# python3 run.py --seed_launch "$seed" --shots_launch -1 --lr_launch "$lr" --model_launch "$model" --dataset_launch kaggle1 --task_launch classifier

# LoRA line
# python3 run.py --seed_launch "$seed" --shots_launch "$shot" --lr_launch "$lr" --iterations 100 --model_launch "$model" --dataset_launch kaggle1 --task_launch lora
# python3 run.py --seed_launch "$seed" --shots_launch "$shot" --lr_launch "$lr" --iterations 100 --model_launch "$model" --dataset_launch hicervix --task_launch lora --level_launch "level_3"

# Percentage analysis
# python3 run.py --seed_launch "$seed" --shots_launch "$shot" --lr_launch "$lr" --iterations 100 --model_launch "$model" --dataset_launch hicervix --level_launch "level_3" --percent_launch 10 --task_launch percentage_lora 
