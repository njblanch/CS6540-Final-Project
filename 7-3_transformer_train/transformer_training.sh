#!/bin/bash

#SBATCH --account=cs6540
#SBATCH --job-name=Ttrain
#SBATCH --output=log/Ttrain_5_output.log
#SBATCH --error=log/Ttrain_5_error.log
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=96GB
#SBATCH --time=24:00:00
#SBATCH --partition=dggpu
#SBATCH --gres=gpu:1
module purge

# Job info
my_job_header

$HOME/miniconda3/envs/dl/bin/python transformer_training.py -v 5
