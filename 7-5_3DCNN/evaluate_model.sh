#!/bin/bash

#SBATCH --account=cs6540
#SBATCH --job-name=3eval
#SBATCH --output=log/3eval_output5.log
#SBATCH --error=log/3eval_error5.log
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=96GB
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
module purge

# Job info
my_job_header

$HOME/miniconda3/envs/videosync3.10/bin/python evaluate_model.py -v e5
