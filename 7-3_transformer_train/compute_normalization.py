import os
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np


def compute_normalization(
    video_path, audio_path, filenames, max_length=225, audio_dim=120, video_dim=1024
):
    audio_sum = torch.zeros(audio_dim)
    audio_sq_sum = torch.zeros(audio_dim)
    video_sum = torch.zeros(video_dim)
    video_sq_sum = torch.zeros(video_dim)
    count = 0

    for filename in tqdm(filenames, desc="Computing normalization parameters"):
        audio_csv = os.path.join(audio_path, f"{filename}.csv")
        video_parquet = os.path.join(video_path, f"{filename}.parquet")

        if not (os.path.exists(audio_csv) and os.path.exists(video_parquet)):
            continue

        try:
            audio_data = pd.read_csv(audio_csv)
            video_data = pd.read_parquet(video_parquet)

            # Group by video_id and clip_number
            audio_groups = audio_data.groupby(["video_id", "video_number"])
            video_groups = video_data.groupby(["video_id", "clip_num"])

            for (video_id, clip_number), audio_group in audio_groups:
                if (f"{video_id}", f"{clip_number}") not in video_groups.groups:
                    continue

                video_group = video_groups.get_group((f"{video_id}", f"{clip_number}"))

                # Process features
                video_features = (
                    video_group.drop(
                        columns=["video_id", "clip_num", "desync", "frame_number"]
                    )
                    .astype(float)
                    .values
                )
                audio_features = (
                    audio_group.drop(
                        columns=["video_id", "video_number", "frame_number"]
                    )
                    .astype(float)
                    .values
                )

                # Pad/truncate audio features to 120
                if audio_features.shape[1] < audio_dim:
                    padding_size = audio_dim - audio_features.shape[1]
                    audio_features = np.pad(
                        audio_features, ((0, 0), (0, padding_size)), "constant"
                    )
                elif audio_features.shape[1] > audio_dim:
                    audio_features = audio_features[:, :audio_dim]

                # Ensure matching sequence lengths
                min_length = min(len(audio_features), len(video_features))
                audio_features = audio_features[:min_length]
                video_features = video_features[:min_length]

                if len(audio_features) == len(video_features):
                    # Convert to tensors
                    audio_tensor = torch.tensor(audio_features, dtype=torch.float)
                    video_tensor = torch.tensor(video_features, dtype=torch.float)

                    # Accumulate sums correctly per feature
                    audio_sum += audio_tensor.sum(dim=0)  # Sum over sequence length
                    audio_sq_sum += (audio_tensor**2).sum(dim=0)
                    video_sum += video_tensor.sum(dim=0)
                    video_sq_sum += (video_tensor**2).sum(dim=0)
                    count += min_length

        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            continue

    if count > 0:
        mean_audio = audio_sum / count
        mean_video = video_sum / count

        # Compute variance and clamp to prevent negative values
        variance_audio = (audio_sq_sum / count - mean_audio**2).clamp(min=1e-6)
        variance_video = (video_sq_sum / count - mean_video**2).clamp(min=1e-6)

        std_audio = variance_audio.sqrt()
        std_video = variance_video.sqrt()

        # Save the normalization parameters
        torch.save(
            {
                "mean_audio": mean_audio,
                "std_audio": std_audio,
                "mean_video": mean_video,
                "std_video": std_video,
            },
            "normalization_params_cae.pth",
        )

        print("Normalization parameters computed and saved.")
    else:
        raise ValueError("No data found for normalization.")


if __name__ == "__main__":
    # Define your paths
    video_path = "/gpfs2/classes/cs6540/AVSpeech/6-1_visual_features/train_1024_cae"
    audio_path = "/gpfs2/classes/cs6540/AVSpeech/5_audio/train"

    # Load filenames (assuming you have a list of filenames)
    with open("all_filenames.txt", "r") as f:
        filenames = [line.strip() for line in f.readlines()]

    compute_normalization(video_path, audio_path, filenames)
