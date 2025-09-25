#!/bin/bash
#SBATCH --job-name=hyena_train
#SBATCH --output=logs/hyena_train_%j.out
#SBATCH --error=logs/hyena_train_%j.err
#SBATCH --account=p31888
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:2
#SBATCH --mem=80G
#SBATCH --time=48:00:00

conda activate deepchopper

# Activate the virtual environment
cd /gpfs/projects/b1171/ylk4626/project/Chimera

# print the current directory
echo "Current directory: $(pwd)"

# print the python version
# Run the training script
uv run train.py experiment=hyena
echo "Training script completed"