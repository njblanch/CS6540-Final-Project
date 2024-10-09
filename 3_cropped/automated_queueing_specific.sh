#!/bin/bash

# Change directories as wanted
base_dir="/gpfs2/classes/cs6540/AVSpeech/2_unzipped"

# Define a list of folder names to process
folders_to_process=("xaa" "xab" "xac" "xad" "xae" "xaf" "xag" "xah")  # Replace with your actual folder names

# Print the list of directories to be processed
echo "Found directories to be processed:"
for dir in "${folders_to_process[@]}"; do
    echo "$dir"
done

# Loop through each model and directory to create and submit job scripts
for dir in "${folders_to_process[@]}"; do
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
#SBATCH --partition=bdgpu
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
