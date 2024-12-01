#!/bin/bash

#SBATCH --account=cs6540
#SBATCH --job-name=sl_3train
#SBATCH --output=log/sl_3train.log
#SBATCH --error=log/sl_3train.log
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=96GB
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
module purge

# Job info
my_job_header

$HOME/miniconda3/envs/videosync3.10/bin/python model_single_load.py -v 5
