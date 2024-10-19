#!/bin/bash

#SBATCH --account=cs6540
#SBATCH --job-name=vis_feats_$1
#SBATCH --output=log/vis_feats_output_$1.log
#SBATCH --error=log/vis_feats_error_$1.log
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --time=12:00:00
#SBATCH --partition=dggpu
#SBATCH --gres=gpu:1
module purge

input_file="./input_list.txt"

while IFS= read -r line; do
    if [[ -z "$line" ]]; then
        continue
    fi
    
    script_loc = "/gpfs2/classes/cs6540/AVSpeech/6_visual_features/visual_features.py"

    /gpfs1/home/g/c/gcercena/miniconda3/envs/dl/bin/python $script_loc -i "$line"


done < "$input_file"
