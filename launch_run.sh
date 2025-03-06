#!/bin/bash
#
#SBATCH --job-name=cyto_finetune
#
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2

source PATH_TO_CHANGE/bin/activate

cd PATH_TO_CHANGE

# Experiment 1: Linear Classifier
# python3 run.py --seed_launch "$seed" --shots_launch -1 --lr_launch "$lr" --model_launch "$model" --dataset_launch kaggle1 --task_launch classifier

# Experiment 2: LoRA Few-Shot Adaptation
# python3 run.py --seed_launch "$seed" --shots_launch "$shot" --lr_launch "$lr" --iterations 100 --model_launch "$model" --dataset_launch kaggle1 --task_launch lora
# python3 run.py --seed_launch "$seed" --shots_launch "$shot" --lr_launch "$lr" --iterations 100 --model_launch "$model" --dataset_launch hicervix --task_launch lora --level_launch "level_3"

# Experiment 3: Pushing Model Fine-Tuning Limits
# python3 run.py --seed_launch "$seed" --shots_launch "$shot" --lr_launch "$lr" --iterations 100 --model_launch "$model" --dataset_launch hicervix --level_launch "level_3" --percent_launch 10 --task_launch percentage_lora 
