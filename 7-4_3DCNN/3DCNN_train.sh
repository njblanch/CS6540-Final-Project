#!/bin/bash

#SBATCH --account=cs6540
#SBATCH --job-name=3train4
#SBATCH --output=log/3train4_output.log
#SBATCH --error=log/3train4_error.log
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=96GB
#SBATCH --time=24:00:00
#SBATCH --partition=dggpu
#SBATCH --gres=gpu:1
module purge

# Job info
my_job_header

$HOME/miniconda3/envs/videosync3.10/bin/python 3DCNN_train.py -v 4
