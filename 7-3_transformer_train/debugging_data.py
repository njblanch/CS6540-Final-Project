import os
import pandas as pd


def debug_sample(filename, video_path, audio_path):
    audio_csv = os.path.join(audio_path, f"{filename}.csv")
    video_parquet = os.path.join(video_path, f"{filename}.parquet")

    if not os.path.exists(audio_csv) or not os.path.exists(video_parquet):
        print(f"Missing files for {filename}")
        return

    audio_data = pd.read_csv(audio_csv)
    video_data = pd.read_parquet(video_parquet)

    audio_groups = set(audio_data.groupby(["video_id", "video_number"]).groups.keys())
    video_groups = set(video_data.groupby(["video_id", "clip_num"]).groups.keys())

    intersection = audio_groups.intersection(video_groups)

    print(f"Filename: {filename}")
    print(f"Audio Groups: {audio_groups}")
    print(f"Video Groups: {video_groups}")
    print(f"Intersection: {intersection}")


# Example usage:
debug_sample("F_XLp-pn-Gc", "/path/to/video", "/path/to/audio")
