#!/bin/bash

#SBATCH --account=cs6540
#SBATCH --job-name=Teval
#SBATCH --output=log/Teval_output.log
#SBATCH --error=log/Teval_error.log
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --time=24:00:00
#SBATCH --partition=dggpu
#SBATCH --gres=gpu:1
module purge

# Job info
my_job_header

$HOME/miniconda3/envs/videosync3.10/bin/python transformer_evaluation.py
