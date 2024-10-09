#!/bin/bash

# Change directories as wanted
base_dir="/gpfs2/classes/cs6540/AVSpeech/2_unzipped"

# List of directories to exclude
# File containing directories to exclude
exclude_file="/gpfs2/classes/cs6540/AVSpeech/3_cropped/exclude_dirs.txt"

# Read excluded directories from the file into an array
mapfile -t exclude_dirs < "$exclude_file"
#exclude_dirs=("xaa" "xab" "xac" "xad" "xae" "xaf" "xag" "xah")

# Get a list of all subdirectories in the specified base directory
dirs=($(find "$base_dir" -mindepth 1 -maxdepth 1 -type d -exec basename {} \;))

# Filter out excluded directories
for exclude in "${exclude_dirs[@]}"; do
    dirs=("${dirs[@]/$exclude}")
done

# Remove any empty entries that may have resulted from filtering
dirs=("${dirs[@]// /}")

# Print the list of directories to be processed
echo "Found directories (after excluding specified ones):"
for dir in "${dirs[@]}"; do
    echo "$dir"
done

# Loop through each model and directory to create and submit job scripts
for dir in "${dirs[@]}"; do
    job_script=$(cat <<EOT
#!/bin/bash

#SBATCH --account=cs6540
#SBATCH --job-name=${dir}
#SBATCH --output=auto_log/${dir}_output.log
#SBATCH --error=auto_log/${dir}_error.log
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --time=48:00:00
#SBATCH --partition=dggpu
#SBATCH --gres=gpu:1
module purge

# Job info
my_job_header

$HOME/miniconda3/envs/videosync3.10/bin/python process_video_clips.py \
/gpfs2/classes/cs6540/AVSpeech/2_unzipped/${dir} \
/gpfs2/classes/cs6540/AVSpeech/3_cropped/train \
/gpfs2/classes/cs6540/AVSpeech/3_cropped/test \
/gpfs2/classes/cs6540/AVSpeech/3_cropped/metadata.jsonl \
15 \
6 \
/gpfs2/classes/cs6540/AVSpeech/3_cropped/avspeech_test.csv
EOT
    )

    # Submit the job
    echo "$job_script" | sbatch
    echo "Submitted job for directory: $dir"

done

