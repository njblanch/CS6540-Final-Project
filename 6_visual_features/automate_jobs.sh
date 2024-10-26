#!/bin/bash

# Check if an input letter is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <letter>"
    echo "Example: $0 a"
    exit 1
fi

# Input letter (e.g., 'a', 'g')
input_letter="$1"

# Fixed initial character
fixed_char="x"

# Master directory prefix based on input
master_prefix="${fixed_char}${input_letter}"

# Base directory containing all 'x<letter><subletter>' directories
base_dir="/gpfs2/classes/cs6540/AVSpeech/2_unzipped"

# Location of the Python script
python_script="/gpfs2/classes/cs6540/AVSpeech/6_visual_features/visual_features.py"

# Conda initialization script and environment
conda_init="/gpfs1/home/g/c/gcercena/miniconda3/etc/profile.d/conda.sh"
conda_env="dl"

# Number of directories per Slurm job
group_size=9

# Output log directory
log_dir="auto_log/${master_prefix}"
mkdir -p "${log_dir}"

# Find all directories starting with 'x<letter><a-z>'
mapfile -t all_dirs < <(find "$base_dir" -mindepth 1 -maxdepth 1 -type d -name "${master_prefix}[a-z]" -exec basename {} \;)

# Check if any directories are found
if [ "${#all_dirs[@]}" -eq 0 ]; then
    echo "No directories found starting with '${master_prefix}' followed by a lowercase letter in '${base_dir}'."
    exit 1
fi

echo "Found ${#all_dirs[@]} directories starting with '${master_prefix}':"
for dir in "${all_dirs[@]}"; do
    echo "  $dir"
done
echo

# Calculate the number of groups
total_dirs=${#all_dirs[@]}
num_groups=$(( (total_dirs + group_size - 1) / group_size ))

# Iterate over each group and submit the Slurm job
for ((g=1; g<=num_groups; g++)); do
    start=$(( (g-1)*group_size ))
    group_dirs=("${all_dirs[@]:start:group_size}")
    
    # Skip empty groups
    if [ "${#group_dirs[@]}" -eq 0 ]; then
        echo "Skipping empty group${g}."
        continue
    fi
    
    # Create a space-separated string of input directories for the Python script
    input_dirs_str=""
    for dir in "${group_dirs[@]}"; do
        input_dirs_str+="\"$base_dir/$dir/$dir\" "
    done
    
    # Define job name
    job_name="group${g}_vis_feats"
    
    # Define log file paths
    output_log="${log_dir}/${job_name}_output.log"
    error_log="${log_dir}/${job_name}_error.log"
    
    # Create the Slurm job script using a heredoc
    job_script=$(cat <<EOT
#!/bin/bash
#SBATCH --account=cs6540
#SBATCH --job-name=${input_letter}_${job_name}
#SBATCH --output=${output_log}
#SBATCH --error=${error_log}
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --partition=dggpu
#SBATCH --gres=gpu:1

module purge

set -x

# Activate the Conda environment
source "${conda_init}"
conda activate "${conda_env}"

# Define the location of the Python script
script_loc="${python_script}"

echo "Executing Python script with command:"
echo "/gpfs1/home/g/c/gcercena/miniconda3/envs/dl/bin/python \"\$script_loc\" -i ${input_dirs_str}"

# Execute the Python script with multiple input directories
/gpfs1/home/g/c/gcercena/miniconda3/envs/dl/bin/python "\$script_loc" -i ${input_dirs_str}
EOT
)

    # Submit the job using sbatch by piping the job script
    echo "----------------------------------------"
    echo "Submitting Group: ${job_name}"
    echo "Directories:"
    for dir in "${group_dirs[@]}"; do
        echo "  $dir"
    done
    echo "Submitting Slurm Job..."
    echo "$job_script" | sbatch
    echo "----------------------------------------"
    echo
done
