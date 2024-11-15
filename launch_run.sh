#!/bin/bash
#
#SBATCH --job-name=VL
#
#SBATCH --cpus-per-task=10
#SBATCH --ntasks=1
#
#SBATCH --mem-per-cpu=10G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#
#SBATCH --mail-type=None
#SBATCH --account=danitim

#SBATCH --output="/gpfs/home/acad/ucl-elen/mdausort/Thyroid/logs/slurmJob_LoRA_shot_%j_%x.out"
#SBATCH --error="/gpfs/home/acad/ucl-elen/mdausort/Thyroid/logs/slurmJob_LoRA_shot_%j_%x.err"

#SBATCH --time=01:00:00
#SBATCH --array=0-0%1

lrs=("1e-2" "3e-3" "1e-3" "3e-4" "1e-4")
seeds=("1" "2" "3")
models=("clip" "uni" "vit_google" "biomedclip" "conch" "quilt")
shots=("0")

index=$SLURM_ARRAY_TASK_ID
lr_idx=$(( index / 18 ))
temp=$(( index % 18 ))
seed_idx=$(( temp / 6 ))
model_idx=$(( temp % 6 ))

lr=${lrs[lr_idx]}
seed=${seeds[seed_idx]}
model=${models[model_idx]}
shot=${shots[0]}

# Updating the output paths to include parameters
output_file="/gpfs/home/acad/ucl-elen/mdausort/Thyroid/logs/slurmJob_LoRA_shot${shot}_seed${seed}_lr${lr}_model${model}.out"
error_file="/gpfs/home/acad/ucl-elen/mdausort/Thyroid/logs/slurmJob_LoRA_shot${shot}_seed${seed}_lr${lr}_model${model}.err"

scontrol update jobid=$SLURM_JOB_ID StdOut=$output_file StdErr=$error_file

module load devel/python/3.9.13
source /gpfs/home/acad/ucl-elen/mdausort/env/clip_lora/bin/activate

cd /gpfs/home/acad/ucl-elen/mdausort/Thyroid/cytology_fine_tuning/

# Classifier line
# python3 run.py --seed_launch "$seed" --shots_launch -1 --lr_launch "$lr" --model_launch "$model" --dataset_launch kaggle1 --task_launch classifier

# LoRA line
# python3 run.py --seed_launch "$seed" --shots_launch "$shot" --lr_launch "$lr" --iterations 100 --model_launch "$model" --dataset_launch kaggle1 --task_launch lora
# python3 run.py --seed_launch "$seed" --shots_launch "$shot" --lr_launch "$lr" --iterations 100 --model_launch "$model" --dataset_launch hicervix --task_launch lora --level_launch "level_3"

# Percentage analysis
python3 run.py --seed_launch "$seed" --shots_launch "$shot" --lr_launch "$lr" --iterations 100 --model_launch "$model" --dataset_launch hicervix --level_launch "level_3" --percent_launch 10 --task_launch percentage_lora 
