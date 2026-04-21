#!/bin/bash -l
# Declaring Slurm Configuration Options

#SBATCH --job-name=short_single_sweep_join_nga7vzky
#SBATCH --account=hic-tts
#SBATCH --partition=tier3
#SBATCH --time=0-12:00:00

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

#SBATCH --mail-user=slack:@jdm8943
#SBATCH --mail-type=ALL

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=a100:1

#SBATCH --mem=16g


# Loading Software/Libraries

source ~/.conda/etc/profile.d/conda.sh
conda activate capstone-model

# Env vars

export WANDB_API_KEY=
export HF_TOKEN=

export SLURM_JOB_LOG="logs/%x_%j.out"

export TOKENIZERS_PARALLELISM=false


# launch sweep agent

SHARED_ARGS="--config style-prompt-generator/short_sweep_config.json \
             --sweep_values style-prompt-generator/short_sweep_values.json \
             --n_folds 5 \
             --count 2 \
             --sweep_id nga7vzky"

# agent 1 creates the sweep and writes the real ID to a file
CUDA_VISIBLE_DEVICES=0 python3 style-prompt-generator/sweep.py $SHARED_ARGS 

