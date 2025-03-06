#!/bin/bash
#
#SBATCH --job-name=clip_lora_us
#
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --qos=preemptible
#
#SBATCH --time=05:00:00
#
#SBATCH --mail-type='FAIL'
#SBATCH --mail-user='manon.dausort@uclouvain.be'
#SBATCH --output="/CECI/proj/medresyst/manon/US/logs/slurmJob_LoRA_shot_%j_%a.out"
#SBATCH --error="/CECI/proj/medresyst/manon/US/logs/slurmJob_LoRA_shot_%j_%a.err"
#


lrs=("5e-3" "1e-4" "5e-4") # --array=0-8%9
seeds=("1")
models=("clip" "biomedclip" "vit_google")
shots=("50")

index=$SLURM_ARRAY_TASK_ID

# Dans ce cas, le produit de (seeds x models x shots) = 1 x 3 x 1 = 3
# Ainsi, on utilise l'index pour sélectionner le lr et le modèle.
lr_idx=$(( index / 3 ))
model_idx=$(( index % 3 ))
seed_idx=0
shot_idx=0

lr=${lrs[lr_idx]}
seed=${seeds[seed_idx]}
model=${models[model_idx]}
shot=${shots[shot_idx]}

source "/CECI/proj/medresyst/histo_cyto/env/tim/bin/activate"

cd "/CECI/proj/medresyst/manon/US/codes/Cytology-fine-tuning/"

# Experiment 1: Linear Classifier
python3 run.py --seed_launch 1 --shots_launch -1 --lr_launch 0.001 --model_launch uni --dataset_launch cu_us1 --task_launch classifier
# python3 run.py --seed_launch "$seed" --shots_launch -1 --lr_launch "$lr" --model_launch "$model" --dataset_launch cu_us2 --task_launch classifier
# python3 run.py --seed_launch "$seed" --shots_launch -1 --lr_launch "$lr" --model_launch "$model" --dataset_launch cu_us3 --task_launch classifier

# Experiment 2: LoRA Few-Shot Adaptation
# python3 run.py --seed_launch "$seed" --shots_launch "$shot" --lr_launch "$lr" --iterations 150 --model_launch "$model" --dataset_launch cu_us1 --task_launch lora
# python3 run.py --seed_launch "$seed" --shots_launch "$shot" --lr_launch "$lr" --iterations 150 --model_launch "$model" --dataset_launch cu_us2 --task_launch lora
# python3 run.py --seed_launch "$seed" --shots_launch "$shot" --lr_launch "$lr" --iterations 150 --model_launch "$model" --dataset_launch cu_us3 --task_launch lora
# python3 run.py --seed_launch "$seed" --shots_launch "$shot" --lr_launch "$lr" --iterations 100 --model_launch "$model" --dataset_launch hicervix --task_launch lora --level_launch "level_3"

# Experiment 3: Pushing Model Fine-Tuning Limits
# python3 run.py --seed_launch "$seed" --shots_launch "$shot" --lr_launch "$lr" --iterations 100 --model_launch "$model" --dataset_launch hicervix --level_launch "level_3" --percent_launch 10 --task_launch percentage_lora 
