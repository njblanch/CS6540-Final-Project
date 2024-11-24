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
def extract_audio_features(video_path, csv_writer, folder_name, video_num, batch_folder):
    # Load the video
    try:
        video = VideoFileClip(video_path)
    except:
        print(f"{batch_folder}: FAILED to read {video_path}", flush=True)
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

        # MFCC Derivatives
        delta_mfcc = librosa.feature.delta(mfcc, mode='nearest')
        delta_mfcc_mean = np.mean(delta_mfcc, axis=1).flatten().tolist()

        delta2_mfcc = librosa.feature.delta(mfcc, order=2, mode='nearest')
        delta2_mfcc_mean = np.mean(delta2_mfcc, axis=1).flatten().tolist()

        # Zero-Crossing Rate (ZCR)
        zcr = librosa.feature.zero_crossing_rate(y=audio_mono)
        zcr_mean = np.mean(zcr).tolist()

        # Spectral Features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_mono, sr=audio_fps)
        spectral_centroid_mean = np.mean(spectral_centroid).tolist()

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_mono, sr=audio_fps)
        spectral_bandwidth_mean = np.mean(spectral_bandwidth).tolist()

        spectral_contrast = librosa.feature.spectral_contrast(y=audio_mono, sr=audio_fps)
        spectral_contrast_mean = np.mean(spectral_contrast, axis=1).flatten().tolist()

        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_mono, sr=audio_fps)
        spectral_rolloff_mean = np.mean(spectral_rolloff).tolist()

        # Root Mean Square Energy (RMS)
        rms = librosa.feature.rms(y=audio_mono)
        rms_mean = np.mean(rms).tolist()

        # Onset Detection
        onset_strength = librosa.onset.onset_strength(y=audio_mono, sr=audio_fps)
        onset_mean = np.mean(onset_strength).tolist()

        # Harmonic-to-Noise Ratio (HNR)
        hnr = librosa.effects.harmonic(y=audio_mono)
        hnr_mean = np.mean(hnr).tolist()

        # Short-Time Energy (STE)
        ste = np.sum(audio_mono ** 2) / len(audio_mono)

        # Mel-Spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=audio_mono, sr=audio_fps, n_mels=64)
        mel_spectrogram_mean = np.mean(mel_spectrogram, axis=1).flatten().tolist()

        # Write the frame number, folder name, video number, and extracted features to the CSV
        csv_writer.writerow([folder_name, video_num, frame_number] + mfcc_mean + delta_mfcc_mean + delta2_mfcc_mean +
                            [zcr_mean, spectral_centroid_mean, spectral_bandwidth_mean] + spectral_contrast_mean + 
                            [spectral_rolloff_mean, rms_mean, onset_mean, hnr_mean, ste] + mel_spectrogram_mean)
        print(f"{batch_folder}: Processed frame {frame_number} of video {video_num} in folder {folder_name} (start: {start_time}, end: {end_time})", flush=True)


# The rest of your code remains unchanged
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
                        print(f"{batch_folder}: Could not delete {file_path}", flush=True)
                        continue
                    print(f"{batch_folder}: Deleted {file_path}", flush=True)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <batch_folder> [override=True/False]", flush=True)
        sys.exit(1)

    # Get the folder passed as a command-line argument
    batch_folder = sys.argv[1]
    
    # Check for override flag (default is False)
    override = len(sys.argv) > 2 and sys.argv[2].lower() == 'true'
    
    # Path to the unzipped directory's subfolder (e.g., xaa, xab)
    batch_folder_path = os.path.join(base_dir, batch_dir, batch_folder)
    
    # Ensure the folder exists
    if not os.path.exists(batch_folder_path):
        print(f"{batch_folder}: Error: Folder {batch_folder_path} does not exist.", flush=True)
        sys.exit(1)

    # List all video folders
    video_folders = []
    for root, _, files in os.walk(batch_folder_path):
        folder_name = os.path.basename(root)
        
        # Check if the folder contains video files
        if any(os.path.join(base_dir, input_dir, file).endswith((".mp4", ".mov", ".avi")) for file in files):
            video_folders.append(folder_name)

    # Check if the batch has already been processed
    if is_batch_completed(batch_folder) or override:
        if override:
            print(f"{batch_folder}: Batch already processed. Deleting output videos due to override.", flush=True)
            delete_output_csvs(video_folders)
        else:
            print(f"{batch_folder}: Batch  already processed. Use override=True to reprocess.", flush=True)
            sys.exit(1)

    if not video_folders:
        print(f"{batch_folder}: No video files found in {batch_folder_path}.", flush=True)
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
                # Write the csv header
                # Define the feature names for the CSV header
                mfcc_feature_names = [f'mfcc_{i}' for i in range(1, 14)]
                delta_mfcc_feature_names = [f'delta_mfcc_{i}' for i in range(1, 14)]
                delta2_mfcc_feature_names = [f'delta2_mfcc_{i}' for i in range(1, 14)]
                spectral_contrast_feature_names = [f'spectral_contrast_{i}' for i in range(1, 8)]
                mel_spectrogram_feature_names = [f'mel_spectrogram_{i}' for i in range(1, 65)]

                feature_names = (
                    mfcc_feature_names +
                    delta_mfcc_feature_names +
                    delta2_mfcc_feature_names +
                    ['zcr', 'spectral_centroid', 'spectral_bandwidth'] +
                    spectral_contrast_feature_names +
                    ['spectral_rolloff'] +
                    ['rms', 'onset_strength', 'hnr', 'ste'] +
                    mel_spectrogram_feature_names
                )

                # Write the CSV header
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
                                extract_audio_features(video_path, writer, folder, video_num, batch_folder)
                            else:
                                print(f"{batch_folder}: Could not extract video number from: {video}. Skipping.", flush=True)

                        else:
                            print(f"{batch_folder}: Could not extract video number from: {video}. Skipping.", flush=True)
                    else:
                        print(f"{batch_folder}: Video {video} not found in {input_dir}. Skipping.", flush=True)


    # Write the completed folder to the text file
    if not is_batch_completed(batch_folder):
        with open(completed_batches_file, "a") as f:
            f.write(f"{batch_folder}\n")

    print(f"{batch_folder}: Completed processing for batch", flush=True)

