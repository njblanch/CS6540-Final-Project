#!/bin/bash

#SBATCH --account=cs6540
#SBATCH --job-name=desync_audio_xa2
#SBATCH --output=log/desync_audio_output_xa2.log
#SBATCH --error=log/desync_audio_error_xa2.log
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=30GB
#SBATCH --time=5:00:00
#SBATCH --partition=dggpu
#SBATCH --gres=gpu:1
module purge

# Job info
my_job_header

# Array of hardcoded inputs
input_array=(
    "xan"
    "xao"
    "xap"
    "xaq"
    "xar"
    "xas"
    "xat"
    "xau"
    "xav"
    "xaw"
    "xax"
    "xay"
    "xaz"
)

# Limit the number of concurrent jobs
max_concurrent_jobs=4
pids=()

# Loop through each element in the array
for line in "${input_array[@]}"; do
    # Run the Python script in the background
    $HOME/miniconda3/envs/videosync3.10/bin/python desync_audio.py "$line" true &
    pids+=($!) # Store the process ID

    # If we've reached the max number of concurrent jobs, wait for one to finish
    while [ "${#pids[@]}" -ge "$max_concurrent_jobs" ]; do
        for pid_index in "${!pids[@]}"; do
            if ! kill -0 "${pids[$pid_index]}" 2>/dev/null; then
                # Remove the finished process from the array
                unset 'pids[pid_index]'
                break
            fi
        done
        # Sleep for a short time before checking again
        sleep 1
    done
done

# Wait for any remaining background jobs to finish
wait
