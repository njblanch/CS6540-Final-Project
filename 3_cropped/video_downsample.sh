#!/bin/bash

#SBATCH --job-name=vtest
#SBATCH --output=log/vtest_output.log
#SBATCH --error=log/vtest_error.log
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
module purge

# Job info
my_job_header

$HOME/miniconda3/envs/videosync3.10/bin/python video_downsample.py
