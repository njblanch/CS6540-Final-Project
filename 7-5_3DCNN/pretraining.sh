#!/bin/bash

#SBATCH --account=cs6540
#SBATCH --job-name=pretraining3
#SBATCH --output=log/pretraining_out.log
#SBATCH --error=log/pretraining_error.log
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=96GB
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
module purge

# Job info
my_job_header

$HOME/miniconda3/envs/videosync3.10/bin/python pretraining.py -v p3
