#!/bin/bash

#SBATCH --account=cs6540
#SBATCH --job-name=vis_feats_$1
#SBATCH --output=log/vis_feats_output_$1.log
#SBATCH --error=log/vis_feats_error_$1.log
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --time=15:00:00
#SBATCH --partition=dggpu
#SBATCH --gres=gpu:1
module purge

# Folder passed as argument
folder_path="/gpfs2/classes/cs6540/AVSpeech/2_unzipped/$1/$1"

# Run the Python script with the folder path as input
$HOME/miniconda3/envs/dl/bin/python visual_features.py -i "$folder_path"
