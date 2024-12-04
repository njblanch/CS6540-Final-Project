#!/bin/bash

#SBATCH --account=cs6540
#SBATCH --job-name=3train1
#SBATCH --output=log/3train_output1.log
#SBATCH --error=log/3train_error1.log
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
module purge

# Job info
my_job_header

$HOME/miniconda3/envs/videosync3.10/bin/python model.py -v f8
