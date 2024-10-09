#!/bin/bash

#SBATCH --account=cs6540
#SBATCH --job-name=vtest
#SBATCH --output=log/vtest_output.log
#SBATCH --error=log/vtest_error.log
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --time=15:00:00
#SBATCH --partition=dggpu
#SBATCH --gres=gpu:1
module purge

# Job info
my_job_header

$HOME/miniconda3/envs/videosync3.10/bin/python process_video_clips.py \
/gpfs2/classes/cs6540/AVSpeech/2_unzipped/xaa \
/gpfs2/classes/cs6540/AVSpeech/3_cropped/train \
/gpfs2/classes/cs6540/AVSpeech/3_cropped/test \
/gpfs2/classes/cs6540/AVSpeech/3_cropped/metadata.jsonl \
15 \
6 \
/gpfs2/classes/cs6540/AVSpeech/3_cropped/avspeech_test.csv

