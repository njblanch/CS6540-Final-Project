from moviepy.editor import VideoFileClip
import os
import csv
import librosa
import numpy as np
import sys
from sklearn.preprocessing import MinMaxScaler

base_dir = "/gpfs2/classes/cs6540/AVSpeech"
batch_dir = "2_unzipped"
input_dir = "4_desynced/train_dist"
output_dir = "5-2_audio/train"
completed_batches_file = os.path.join(base_dir, "5-2_audio", "completed_batches.txt")

# Function to extract audio features and save them to the same CSV for videos with the same prefix
def extract_audio_features(video_path, csv_writer, folder_name, video_num, batch_folder):
    # Load the video
    try:
        video = VideoFileClip(video_path)
    except:
        print(f"{batch_folder}: FAILED to read {video_path}", flush=True)
        return

    # Load the audio from the video
    audio = video.audio
    if audio is None:
        print(f"{batch_folder}: No audio found in {video_path}", flush=True)
        return

    sr = 16000                # Audio sampling rate

    # Extract the audio as samples at the default sample rate
    audio_samples = audio.to_soundarray(fps=sr)

    print(f"{folder_name} - {video_num} Audio Samples: {len(audio_samples)}", flush=True)
    
    # Convert stereo to mono if needed
    if audio_samples.ndim > 1:
        audio_samples = np.mean(audio_samples, axis=1)

    # Compute the MFCCs with a Hamming window of size 256
    n_fft = 304               # Window size
    hop_length = 152          # Overlap (50% of n_fft)
    n_mfcc = 13               # Number of MFCCs to compute

    # Calculate MFCCs with the specified Hamming window
    mfccs = librosa.feature.mfcc(y=audio_samples, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, window='hamming', center=False)

    # Drop the first MFCC, as it generally represents overall signal energy
    mfccs = mfccs[1:]

    # Normalize MFCCs to the range [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    mfccs_normalized = scaler.fit_transform(mfccs.T).T

    # Write MFCC features to CSV
    num_rows = 0
    for mfcc in mfccs_normalized.T:
        csv_writer.writerow([folder_name, video_num] + list(mfcc))
        num_rows += 1

    print(f"{folder_name} - {video_num} Num Rows: {num_rows}", flush=True)
    print(f"{batch_folder}: Processed {video_path}", flush=True)


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
                feature_names = [f'mfcc_{i}' for i in range(2, 14)]

                # Write the CSV header
                writer.writerow(['video_id', 'video_number'] + feature_names)

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

