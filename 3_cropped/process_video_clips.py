import os
import pandas as pd
from video_downsample import downsample_with_moviepy_no_saving
from isolate_heads import process_video_videofile
import cv2
import argparse
from moviepy.editor import VideoFileClip, ImageSequenceClip, CompositeVideoClip
import json

# NOTE: Moviepy just keeps wanting to print, even when being suppressed. Fix this if necessary

# NOTE: Lot of exceptions since FFMPEG can be buggy


def load_metadata(metadata_path):
    metadata = {}

    with open(metadata_path, 'r') as file:
        for line in file:
            entry = json.loads(line)
            video_id = entry['metadata']['video_id']
            frames = entry['frames']
            metadata[video_id] = frames

    return metadata


if __name__ == "__main__":
    # CLI arguments and load filenames
    parser = argparse.ArgumentParser(description='List all files in a specified directory.')
    parser.add_argument('in_dir', type=str, help='Path to input directory')
    parser.add_argument('train_out_dir', type=str, help='Path to train output directory')
    parser.add_argument('test_out_dir', type=str, help='Path to test output directory')
    parser.add_argument('metadata', type=str, help='Path to metadata JSONL file')
    parser.add_argument('fps', type=int, help='Target FPS')
    parser.add_argument('min_duration', type=int, help='Minimum duration of video')
    parser.add_argument('test_csv', type=str, help='Filename for test csv')

    args = parser.parse_args()

    print(f"Input directory: {args.in_dir}", flush=True)

    test_df = pd.read_csv(args.test_csv, sep='\t', header=None, names=['filename', 'start_segment', 'end_segment', 'start_x_pct', 'start_y_pct'])
    test_ids = set(test_df['filename'].tolist())

    metadata = load_metadata(args.metadata)

    # Load file for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    prnt_accum = 0

    for subdir, _, files in os.walk(args.in_dir):
        # Simply for keeping track of progress on Vacc
        if prnt_accum % 25 == 0:
            print(f"{prnt_accum} videos processed", flush=True)
        prnt_accum += 1

        subdir_id = os.path.basename(subdir)
        if subdir_id in test_ids:
            output_dir = args.test_out_dir
            print("test_dir", flush=True)
        else:
            output_dir = args.train_out_dir

        video_files = sorted(files)

        for index, file in enumerate(video_files):
            filename = os.path.splitext(file)[0]
            video_id, clip_number = filename.rsplit('_', 1)

            try:
                clip_number = int(clip_number)
            except ValueError:
                # print(f"Invalid clip number for file {file}, skipping.")
                continue

            if video_id not in metadata:
                # print(f"No data found for {video_id}, skipping.")
                continue
            # print(clip_number, flush=True)
            frames = metadata[video_id]

            # Load and downsample video - also checks for video length
            # input_file_path = os.path.join(args.in_dir, file)
            input_file_path = os.path.join(subdir, file)
            try:
                downsampled_video = downsample_with_moviepy_no_saving(input_file_path, target_fps=args.fps,
                                                                      min_duration=args.min_duration)
            except Exception as e:
                continue

            # Skip short videos
            if downsampled_video is None:
                # print("Too short")
                continue

            if clip_number < len(metadata[video_id]):
                frame = metadata[video_id][clip_number]
                start_x_pct = frame['x']
                start_y_pct = frame['y']
                # print(start_x_pct, start_y_pct)

                # Convert percentage to pixel values
                width, height = downsampled_video.size
                start_x = int(start_x_pct * width)
                start_y = int(start_y_pct * height)
                # print(start_x, start_y)

                # Head-crop video
                try:
                    processed_frames = process_video_videofile(start_x, start_y, downsampled_video, face_cascade)
                except Exception as e:
                    continue

                if processed_frames is None or not processed_frames:
                    # print("No face found")
                    continue
                try:
                    cropped_video = ImageSequenceClip(processed_frames, fps=args.fps)

                    final_video = CompositeVideoClip([cropped_video.set_audio(downsampled_video.audio)])

                    # Save the cropped video with audio
                    output_file_path = os.path.join(output_dir, f'{filename}.mp4')  # Specify output filename
                    final_video.write_videofile(output_file_path, codec='libx264', fps=args.fps, verbose=False, write_logfile=False, remove_temp=True) # remove temporary audio files
                except Exception as e:
                    pass

    print("########################\nDone!\n########################")

