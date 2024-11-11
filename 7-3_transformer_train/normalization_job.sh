#!/bin/bash

#SBATCH --account=cs6540
#SBATCH --job-name=normalization
#SBATCH --output=log/normalization.out
#SBATCH --error=log/normalization.err
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=96GB
#SBATCH --time=24:00:00
#SBATCH --partition=bluemoon
module purge

# Job info
my_job_header

$HOME/miniconda3/envs/dl/bin/python compute_normalization.py
