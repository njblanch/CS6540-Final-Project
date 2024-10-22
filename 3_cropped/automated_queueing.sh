#!/bin/bash

# find /gpfs2/classes/cs6540/AVSpeech/3_cropped/train_dist -maxdepth 1 -type f | wc -l

# Directories already run:
# xa*, xb*, xc*, xd*, xe*, xf*, xg*, xh*
# xi* running

# Change directories as wanted
base_dir="/gpfs2/classes/cs6540/AVSpeech/2_unzipped"

# List of directories to exclude
# File containing directories to exclude
exclude_file="exclude_dirs.txt"

# Read excluded directories from the file into an array
#mapfile -t exclude_dirs < "$exclude_file"
exclude_dirs=("xaa" "xab" "xac" "xad" "xae" "xaf" "xag" "xah")

echo "Excluded directories:"
for exclude in "${exclude_dirs[@]}"; do
    echo "$exclude"
done

# Get a list of all subdirectories in the specified base directory
dirs=($(find "$base_dir" -mindepth 1 -maxdepth 1 -type d -exec basename {} \;))


filtered_dirs=()
for dir in "${dirs[@]}"; do
    # Check if the directory starts with 'xa'
    if [[ "$dir" == xi* ]]; then
        # Check if the directory is not in the exclude list
        filtered_dirs+=("$dir")
    fi
done

# Filter out excluded directories
for exclude in "${exclude_dirs[@]}"; do
    filtered_dirs=("${filtered_dirs[@]/$exclude}")
done

# Remove any empty entries that may have resulted from filtering
filtered_dirs=("${filtered_dirs[@]// /}")

non_empty_dirs=()
for dir in "${filtered_dirs[@]}"; do
    if [[ -n "$dir" ]]; then  # Check if the entry is non-empty
        non_empty_dirs+=("$dir")
    fi
done

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
#SBATCH --job-name=${dir}
#SBATCH --output=auto_log/${dir}_output.log
#SBATCH --error=auto_log/${dir}_error.log
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --time=24:00:00
#SBATCH --partition=dggpu
#SBATCH --gres=gpu:1
module purge

# Job info
my_job_header

$HOME/miniconda3/envs/videosync3.10/bin/python process_video_clips.py \
/gpfs2/classes/cs6540/AVSpeech/2_unzipped/${dir} \
/gpfs2/classes/cs6540/AVSpeech/3_cropped/train_dist \
/gpfs2/classes/cs6540/AVSpeech/3_cropped/test_dist \
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

