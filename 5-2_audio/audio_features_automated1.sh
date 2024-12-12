#!/bin/bash

#SBATCH --account=cs6540
#SBATCH --job-name=audio_features_xp1
#SBATCH --output=log/audio_features_output_xp1.log
#SBATCH --error=log/audio_features_error_xp1.log
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=40GB
#SBATCH --time=12:00:00
#SBATCH --partition=bluemoon
module purge

# Job info
my_job_header

# Automate the creation of the input_array
input_array=()
prefix="xp"
letters=("a" "b" "c" "d" "e" "f" "g" "h" "i" "j" "k" "l" "m")

# Generate input_array with the desired pattern
for letter in "${letters[@]}"; do
    input_array+=("${prefix}${letter}")
done

# Limit the number of concurrent jobs
max_concurrent_jobs=4
pids=()

# Loop through each element in the array
for line in "${input_array[@]}"; do
    # Run the Python script in the background
    $HOME/miniconda3/envs/videosync3.10/bin/python audio_features_new.py "$line" true &
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
