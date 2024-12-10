#!/bin/bash

#SBATCH --account=cs6540
#SBATCH --job-name=3train_faster52
#SBATCH --output=log/3train_output_faster52.log
#SBATCH --error=log/3train_error_faster52.log
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
module purge

# Job info
my_job_header

$HOME/miniconda3/envs/videosync3.10/bin/python model_faster.py -v faster52
