# CS6540-Final-Project
Nathan Blanchard, Skyler Heininger, Gian Cercena

## Description

The issue our project attempts to handle is the detection of desynchronization between video and audio in videos where an individual’s face is shown, and more specifically, the amount it is desynced by. This is an important problem to solve for various reasons. Many videos, for a variety of reasons, can have their audio and video mistimed. Whether it be the corruption of a file, or an old analog video recording that naturally had its video desynced, this project has many opportunities to help people. From families restoring old home videos, to museums restoring large amounts of old footage, instead of having an individual sit through and adjust each video, massive amounts of time can be saved by shifting this problem over to an automated system. Potential improvements could even lead to systems such as these happening in near-real-time, allowing for even another space of improvement.

## Algorithm Description

Our first pre-processing step starts with the cropping of the video using OpenCV’s headless model, which performs object detection and cropping of the video to just encapsulate the person’s face. We have measures in place to reject videos where OpenCV’s model finds no face present. The threshold is a face not being detected concurrently for a certain amount of frames. We additionally reject videos where the face moves by too many pixels in a video. While this can occur naturally in videos, this generally happens when there are multiple faces close together in the video, causing the algorithm to sometimes select the wrong face as the target. Thus, we reject these videos using a distance measure between faces in consecutive frames.

Subsequently, we desynchronize the video and audio. We shift the audio +- 1 second, chosen uniformly. Note that we also reduce the framerate of each video down to 15fps, meaning that the audio could be up to 15 frames off.

Once we have desynced cropped videos, our process diverges into two branches where each extracts features from each frame of all videos. The first looks at audio. Since audio isn’t split into frames like video is, we manually divide the audio into chunks and then get features from each. Using the Python package Librosa, we collect features that are tailored for speech processing, such as MFCC, ZCR, and Mel-Spectrogram features, allowing us to capture essential characteristics of spoken language. This step in audio processing outputs 118 features per segment of audio corresponding to a single frame of video.

The second branch looks at the video. We perform feature extraction using a pre-trained CNN. The CNN chosen was EfficientNetV2 due to its quick processing, high accuracy, and reputation, which will be noted on later. The model outputs 1280 features per frame, which can be reduced via an autoencoder if decided.

The visual and audio features are then concatenated together to create the input to the transformer. The label for each video is the amount of frames the audio is desynced, ranging from -15 to +15. The transformer uses this input to predict a single desynchronization value.


## Steps to Reproduce

1. Download and unzip files from the [AVSpeech dataset](https://looking-to-listen.github.io/avspeech/)
2. Use a command line utility to extract the zipped files to 2_unzipped/.
3. `3_cropped/process_video_clips.py`: This will first downsample videos to a framerate of 15 frames per second, which will increase processing speed and make the size of our data more feasible to process. Next, it will isolate the head of the person speaking in the video, cropping to a bounding box of size 256x256.
4. `4_desynced/desync_audio.py`: This script trims the video by 1 second on either end, preserving audio. This audio clip is then placed at a random location on the video, creating a desynchronization of +/- 15 frames, or one second, between the audio and the video. The amount of desync is then appended to the video name and the video is re-saved in 5_audio. Note that this step of processing is limited to video clips that are 6+ seconds long.
5. `5-2_audio/audio_features_new.py`: This step extracts audio features, saving all feature observations of the same video id to a single CSV to make the train-test split easier to perform without accidentally having training data leak into our test data. To identify audio features that correspond to which video clip, this has rows `video_id` and `video_number`. These features include MFCC audio features, commonly used in speech processing tasks. There are multiple rows of audio feature readings per video frame.
6. `6-2_visual_features/visual_features_lip_and_optical.py`: This uses the Gunnar Farneback algorithm to calculate the vertical pixel level velocities between frames of video. This was applied to just the lip region of each face in a frame, utilizing InsightFace’s RetinaFace model to determine the location of the lips in each frame. This also contains rows ‘video_id’, ‘clip_number’, and ‘frame_number’. Alternative methods for generating audio features can be found in 6_visual_features and 6-1_visual features, which uses the last hidden layer of ENV2 and a trained CNN autoencoder, respectively. The code for training the autoencoder can be found in 6-1_visual_features.
8. `7-5_3DCNN/model_faster.py`: This script contains our implementation of model 3, specifically with vertical optical flows. Important to note, this script will resynchronize the data during loading, and then perform three desynchronizations randomly. This is considered “faster” since the previous alternative in the same folder performs a sliding window over all possible desynchronization values. Scripts for training older models exist in 7-3_transformer_train/ transformer_training.py (model 1), 7-LSTM (model 2), and 7-4_3DCNN/ 3DCNN_train.py (model 3 with auto-encoded features).
9. `7-5_3DCNN/evaluate_model.py`: This provides all the necessary code to replicate figures. Similar scripts exist in other model directories for replicating results, however, only differ in data loaders and model imports.


## VACC Information

### Starting Jobs

See 3_cropped/process_videos.sh for an example - you may need to change file paths, etc, but the important bit is the #!/bin/bash at the start and the #SBATCH commands.

These specify the resources for your job, partition to run on, output and error logs, and job name (useful for canceling). After the #SBATCH commands, you are in CLI and can call anything you'd like. In process_videos.sh there is a call to python within miniconda3, which you may need to change for your purposes/name of the environment. To note, change the partition to "dggpu" for deepgreen, "bdgpu" for blackdiamond, and "bluemoon" for bluemoon partition. See knowledge base for more info.


### List of Useful Commands

* Queue job: `sbatch batch_file.sh`

* Check queued jobs: `squeue -u username`

* View jobs running on a partition: `squeue -p partition_name`

* View output log: `cat log/log_name_output.log`, `cat log/log_name_error.log`

* Cancel all jobs: `scancel -u username`

* Cancel specific job: `scancel job_id`, `scancel -n job_name`

* Check number of files in a directory:  `ls /path/to/directory -1 | wc -l`
