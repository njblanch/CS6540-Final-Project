#!/bin/bash

#SBATCH --account=cs6540
#SBATCH --job-name=vis_feats
#SBATCH --output=log/vis_feats_output.log
#SBATCH --error=log/vis_feats_error.log
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --time=15:00:00
#SBATCH --partition=dggpu
#SBATCH --gres=gpu:1
module purge

# Job info
# get_vis_feats

# Read inputs from a file
input_file="/gpfs2/classes/cs6540/AVSpeech/6_visual_features/input_list.txt"

$HOME/miniconda3/envs/dl/bin/python visual_features.py -i /gpfs2/classes/cs6540/AVSpeech/2_unzipped/xaa/xaa/

# Loop through each line in the input file
# while IFS= read -r line; do
#     $HOME/miniconda3/envs/dl/bin/python visual_features.py -i /gpfs2/classes/cs6540/AVSpeech/2_unzipped/$line/$line/ &
# done < "$input_file"

# Wait for all background jobs to finish
wait
