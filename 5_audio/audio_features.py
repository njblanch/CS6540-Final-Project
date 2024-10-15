from moviepy.editor import VideoFileClip
import os
import csv
import librosa
import numpy as np
import sys

base_dir = "/gpfs2/classes/cs6540/AVSpeech"
batch_dir = "2_unzipped"
input_dir = "4_desynced/train_dist"
output_dir = "5_audio/train"
completed_batches_file = os.path.join(base_dir, "5_audio", "completed_batches.txt")

# Function to extract audio features and save them to the same CSV for videos with the same prefix
def extract_audio_features(video_path, csv_writer, folder_name, video_num):
    # Load the video
    try:
        video = VideoFileClip(video_path)
    except:
        print(f"{batch_folder}: FAILED to read {video_path}")
        return
    
    # Get the video properties
    total_frames = video.reader.nframes
    fps = video.fps
    duration = video.duration

    # Load the audio from the video
    audio = video.audio

    # Loop through each frame and extract the corresponding audio features
    for frame_number in range(total_frames):
        # Calculate the start and end time for each frame
        start_time = frame_number / fps
        end_time = (frame_number + 1) / fps if frame_number < total_frames - 1 else duration

        # Deal with some errors
        if end_time > duration:
            end_time = duration
        if start_time == end_time:
            break
        
        # Extract the audio segment corresponding to the frame's timestamps
        audio_subclip = audio.subclip(start_time, end_time)

        audio_fps = 44100
        audio_data = audio_subclip.to_soundarray(fps=audio_fps)

        # Convert to mono
        audio_mono = np.mean(audio_data, axis=1)

        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio_mono, sr=audio_fps, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1).flatten().tolist()

        # Zero-Crossing Rate (ZCR)
        zcr = librosa.feature.zero_crossing_rate(y=audio_mono)
        zcr_mean = np.mean(zcr).tolist()

        # Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_mono, sr=audio_fps)
        spectral_centroid_mean = np.mean(spectral_centroid).tolist()

        # Root Mean Square Energy (RMS)
        rms = librosa.feature.rms(y=audio_mono)
        rms_mean = np.mean(rms).tolist()

        # Onset Detection
        onset_strength = librosa.onset.onset_strength(y=audio_mono, sr=audio_fps)
        onset_mean = np.mean(onset_strength).tolist()

        # Write the frame number, folder name, video number, and extracted features to the CSV
        csv_writer.writerow([folder_name, video_num, frame_number] + mfcc_mean + [zcr_mean, spectral_centroid_mean, rms_mean, onset_mean])
        print(f"Processed frame {frame_number} of video {video_num} in folder {folder_name} (start: {start_time}, end: {end_time})")


# Function to check if the batch has already been processed
def is_batch_completed(batch_name):
    if os.path.exists(completed_batches_file):
        with open(completed_batches_file, "r") as f:
            completed_batches = f.read().splitlines()
        return batch_name in completed_batches
    return False

# Function to delete processed videos for a batch based on video names in videos_to_process
def delete_output_csvs(csv_names):
    for root, _, files in os.walk(os.path.join(base_dir, output_dir)):
        for file in files:
            # Check if the file in output folder starts with any of the csv names
            for name in csv_names:
                if file.startswith(os.path.splitext(name)[0]):
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                    except:
                        print(f"Could not delete {file_path}")
                        continue
                    print(f"Deleted {file_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <batch_folder> [override=True/False]")
        sys.exit(1)

    # Get the folder passed as a command-line argument
    batch_folder = sys.argv[1]
    
    # Check for override flag (default is False)
    override = len(sys.argv) > 2 and sys.argv[2].lower() == 'true'
    
    # Path to the unzipped directory's subfolder (e.g., xaa, xab)
    batch_folder_path = os.path.join(base_dir, batch_dir, batch_folder)
    
    # Ensure the folder exists
    if not os.path.exists(batch_folder_path):
        print(f"Error: Folder {batch_folder_path} does not exist.")
        sys.exit(1)

    # List all video folders
    video_folders = []
    for root, _, files in os.walk(batch_folder_path):
        folder_name = os.path.basename(root)
        
        # Check if the folder contains video files
        if any(os.path.join(base_dir, input_dir, file).endswith((".mp4", ".mov", ".avi")) for file in files):
            video_folders.append(folder_name)

    # Check if the batch has already been processed
    if is_batch_completed(batch_folder):
        if override:
            print(f"Batch {batch_folder} already processed. Deleting output videos due to override.")
            delete_output_csvs(video_folders)
        else:
            print(f"Batch {batch_folder} already processed. Use override=True to reprocess.")
            sys.exit(1)

    if not video_folders:
        print(f"No video files found in {batch_folder_path}.")
        sys.exit(1)

    # Process videos from input_dir that match the video prefixes
    output_path = os.path.join(base_dir, output_dir)
    video_extensions = (".mp4", ".mov", ".avi")
    for folder in video_folders:
        contains_videos = False
        video_names = []
        for file in os.listdir(os.path.join(base_dir, input_dir)):
            if file.startswith(folder) and file.endswith(video_extensions):
                if not contains_videos:
                    contains_videos = True
                video_names.append(file)

        if contains_videos:
            # Prepare a single CSV file for this folder
            csv_file = os.path.join(base_dir, output_path, f"{folder}.csv")
            with open(csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                # Write the header (folder name, video number, frame number + feature names)
                mfcc_feature_names = [f'mfcc_{i}' for i in range(1, 14)]
                feature_names = mfcc_feature_names + ['zcr', 'spectral_centroid', 'rms', 'onset_strength']
                writer.writerow(['video_id', 'video_number', 'frame_number'] + feature_names)

                # Process each video with the matching folder
                for video in video_names:
                    video_path = os.path.join(base_dir, input_dir, video)
                    if os.path.exists(video_path):
                        # Extract the number between the last occurrence of '_d' and the preceding '_'
                        last_d_index = video.rfind('_d')
                        if last_d_index != -1:  # Ensure '_d' was found
                            # Find the preceding underscore before '_d'
                            preceding_underscore_index = video.rfind('_', 0, last_d_index)
                            if preceding_underscore_index != -1:  # Ensure there's an underscore before '_d'
                                # Extract the number between the two indices
                                video_num = video[preceding_underscore_index + 1:last_d_index]
                                extract_audio_features(video_path, writer, folder, video_num)
                            else:
                                print(f"Could not extract video number from: {video}. Skipping.")

                        else:
                            print(f"Could not extract video number from: {video}. Skipping.")
                    else:
                        print(f"Video {video} not found in {input_dir}. Skipping.")


    # Write the completed folder to the text file
    if not is_batch_completed(batch_folder):
        with open(completed_batches_file, "a") as f:
            f.write(f"{batch_folder}\n")

    print(f"Completed processing for batch: {batch_folder}")
