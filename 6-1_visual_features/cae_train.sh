#!/bin/bash
#SBATCH --account=cs6540
#SBATCH --job-name=cae_train
#SBATCH --output=log/cae_train_output.log
#SBATCH --error=log/cae_train_error.log
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
my_job_header

module purge

$HOME/miniconda3/envs/videosync3.10/bin/python cae_train.py