#!/bin/bash
#SBATCH -t 48:00:00                        # Time limit (hh:mm:ss)
#SBATCH --account=p31888                   # Account name
#SBATCH --partition=gengpu                 # Partition name
#SBATCH --mem=50G                          # RAM
#SBATCH --gres=gpu:h100:4                  # Number of GPUs
#SBATCH --constraint=rhel8
#SBATCH --ntasks-per-node=4                # Should correspond to num devices (at least 1-1 task to GPU)
#SBATCH --cpus-per-task=4                  # Increased CPU cores per task for better parallelization
#SBATCH -N 1                               # Number of nodes
#SBATCH --job-name=mambasp_optuna          # Job name
#SBATCH --output=./slurm_log/%x_%j.log     # Log file
#SBATCH --export=all                        # ensures that all environment variables from the submitting shell

# Print environment information for debugging
echo "current directory: $(pwd)"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "GPU information: $(nvidia-smi)"

# Run the training with distributed data parallel
uv run train.py -m hparams_search=mambasp_optuna experiment=mambasp
