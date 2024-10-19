#!/bin/bash

#SBATCH --account=cs6540
#SBATCH --job-name=vis_feats
#SBATCH --output=log/vis_feats_output.log
#SBATCH --error=log/vis_feats_error.log
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
    
    script_loc="/gpfs2/classes/cs6540/AVSpeech/6_visual_features/visual_features.py"

    video_src="/gpfs2/classes/cs6540/AVSpeech/2_unzipped/$line/$line/"

    /gpfs1/home/g/c/gcercena/miniconda3/envs/dl/bin/python $script_loc -i $video_src &

done < "$input_file"

wait