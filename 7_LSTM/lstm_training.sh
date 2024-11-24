#!/bin/bash

#SBATCH --account=cs6540
#SBATCH --job-name=lstm_training
#SBATCH --output=log/lstm_training_output.log
#SBATCH --error=log/lstm_training_error.log
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=96GB
#SBATCH --time=24:00:00
#SBATCH --partition=dggpu
module purge

# Job info
my_job_header

$HOME/miniconda3/envs/videosync3.10/bin/python lstm_training_new.py -v 1

# Wait for any remaining background jobs to finish
wait
