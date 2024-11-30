import os
import random
import sys
from moviepy.editor import VideoFileClip
import time

FPS = 15
MAX_SHIFT_FRAMES = 30
base_dir = "/gpfs2/classes/cs6540/AVSpeech"
batch_dir = "2_unzipped"
input_dir = "3_cropped/train_dist"
output_dir = "4_desynced/train_dist"
completed_batches_file = os.path.join(base_dir, "4_desynced", "completed_batches_dist.txt")

# Function to trim video and shift audio
def desync_audio(video_path, batch_folder):
    # Load video
    try:
        clip = VideoFileClip(video_path)
    except:
        print(f"{batch_folder}: Error reading file, trying again {video_path}", flush=True)
        try: 
            time.sleep(1)
            clip = VideoFileClip(video_path)
        except:
            print(f"{batch_folder}: FAILED to read {video_path}", flush=True)
            return
    video_duration = clip.duration  # In seconds

    # 1. Trim video by one second on both sides
    start_trim = (MAX_SHIFT_FRAMES / 2) / FPS
    end_trim = (MAX_SHIFT_FRAMES / 2) / FPS
    trimmed_clip = clip.subclip(start_trim, video_duration-end_trim)

    # 2. Shift the audio by a random number of frames
    trim_frames = random.randint(0, MAX_SHIFT_FRAMES)
    trim_seconds = trim_frames / FPS

    # Trim audio track
    audio_trimmed = clip.audio.subclip(trim_seconds, video_duration-((start_trim+end_trim)-trim_seconds))

    # 3. Stitch the shifted audio and trimmed video together
    desynced_clip = trimmed_clip.set_audio(audio_trimmed)

    # 4. Save the desynced video
    video_filename = os.path.basename(video_path)
    new_filename = f"{os.path.splitext(video_filename)[0]}_d_{-(int(MAX_SHIFT_FRAMES/2-trim_frames))}.mp4"
    output_path = os.path.join(base_dir, output_dir, new_filename)
    
    try:
        desynced_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
    except:
        print(f"{batch_folder}: Error writing file, trying again {output_path}", flush=True)
        try:
            time.sleep(1)
            desynced_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
        except:
            print(f"{batch_folder}: FAILED to write {output_path}", flush=True)
            return

# Function to check if the batch has already been processed
def is_batch_completed(batch_name):
    if os.path.exists(completed_batches_file):
        with open(completed_batches_file, "r") as f:
            completed_batches = f.read().splitlines()
        return batch_name in completed_batches
    return False

# Function to delete processed videos for a batch based on video names in videos_to_process
def delete_output_videos(videos_to_process):
    for root, _, files in os.walk(os.path.join(base_dir, output_dir)):
        for file in files:
            # Check if the file in output folder starts with any of the video names
            for video in videos_to_process:
                if file.startswith(os.path.splitext(video)[0]):
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

    # List all videos in the specified batch subfolder and its subfolders
    videos_to_process = []
    for root, _, files in os.walk(batch_folder_path):
        for file in files:
            if file.endswith((".mp4", ".mov", ".avi")):
                videos_to_process.append(file)

    if not videos_to_process:
        print(f"{batch_folder}: No video files found in {batch_folder_path}.", flush=True)
        sys.exit(1)

    # Check if the batch has already been processed
    if is_batch_completed(batch_folder) or override:
        if override:
            print(f"{batch_folder}: Batch already processed. Deleting output videos due to override.", flush=True)
            delete_output_videos(videos_to_process)
        else:
            print(f"{batch_folder}: Batch already processed. Use override=True to reprocess.", flush=True)
            sys.exit(1)

    # Process videos from 'downloaded_segments' that match the video names
    for video in videos_to_process:
        video_path = os.path.join(base_dir, input_dir, video)
        if os.path.exists(video_path):
            desync_audio(video_path, batch_folder)
        else:
            print(f"{batch_folder}: Video {video} not found in {input_dir}. Skipping.", flush=True)

    # Write the completed folder to the text file
    if not is_batch_completed(batch_folder):
        with open(completed_batches_file, "a") as f:
            f.write(f"{batch_folder}\n")

    print(f"{batch_folder}: Completed processing for batch", flush=True)
