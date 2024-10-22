#!/bin/bash

master_dir="xa"

# Change directories as wanted
base_dir="/gpfs2/classes/cs6540/AVSpeech/2_unzipped"

# Get a list of all subdirectories in the specified base directory
dirs=($(find "$base_dir" -mindepth 1 -maxdepth 1 -type d -exec basename {} \;))


filtered_dirs=()
for dir in "${dirs[@]}"; do
    # Check if the directory starts with master_dir*
    if [[ "$dir" == "$master_dir"* ]]; then
        # Check if the directory is not in the exclude list
        filtered_dirs+=("$dir")
    fi
done

# Remove any empty entries that may have resulted from filtering
filtered_dirs=("${filtered_dirs[@]// /}")

non_empty_dirs=()
for dir in "${filtered_dirs[@]}"; do
    if [[ -n "$dir" ]]; then  # Check if the entry is non-empty
        non_empty_dirs+=("$dir")
    fi
done

# keeping only first 3 in non_empty_dirs for debugging
non_empty_dirs=("${non_empty_dirs[@]:0:3}")


# Print the list of directories to be processed
echo "Found directories (after excluding specified ones):"
for filtered_dir in "${non_empty_dirs[@]}"; do
    echo "$filtered_dir"
done

# Loop through each model and directory to create and submit job scripts
for dir in "${non_empty_dirs[@]}"; do
    job_script=$(cat <<EOT
#!/bin/bash

#SBATCH --account=cs6540
#SBATCH --job-name=${dir}_vis_feats
#SBATCH --output=auto_log/${master_dir}/${dir}_output.log
#SBATCH --error=auto_log/${master_dir}/${dir}_error.log
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --time=4:00:00
#SBATCH --partition=dggpu
#SBATCH --gres=gpu:1
module purge

# Job info
my_job_header

script_loc="/gpfs2/classes/cs6540/AVSpeech/6_visual_features/visual_features.py"
video_src="/gpfs2/classes/cs6540/AVSpeech/2_unzipped/${dir}/${dir}/"

/gpfs1/home/g/c/gcercena/miniconda3/envs/dl/bin/python $script_loc -i $video_src
EOT
)

    # Submit the job
    echo "$job_script" | sbatch
    echo "Submitted job for directory: $dir"

done

