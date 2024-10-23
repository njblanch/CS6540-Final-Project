#!/bin/bash

#SBATCH --account=cs6540
#SBATCH --job-name=desync_audio
#SBATCH --output=log/desync_audio_output.log
#SBATCH --error=log/desync_audio_error.log
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --time=15:00:00
#SBATCH --partition=bdgpu
#SBATCH --gres=gpu:1
module purge

# Job info
my_job_header

# Read inputs from a file
input_file="/users/n/j/njblanch/videosync/audio_features/input_list.txt"

# Loop through each line in the input file
while IFS= read -r line; do
    $HOME/miniconda3/envs/videosync3.10/bin/python audio_features.py "$line" true &
done < "$input_file"

# Wait for all background jobs to finish
wait