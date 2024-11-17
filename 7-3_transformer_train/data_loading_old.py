# data_loading.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import logging

OFFSET = 15

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


class VideoAudioDataset(Dataset):
    def __init__(
        self,
        filenames,
        video_path,
        audio_path,
        normalization_params,
        max_length=225,
        normalize=True,
        cache_size=1000,
    ):
        """
        Args:
            filenames (list): List of filenames without extensions.
            video_path (str): Path to video feature files (.parquet).
            audio_path (str): Path to audio feature files (.csv).
            normalization_params (dict): Precomputed normalization parameters.
            max_length (int): Maximum sequence length for padding/truncation.
            normalize (bool): Whether to normalize features.
            cache_size (int): Number of samples to cache in memory.
        """
        self.filenames = filenames
        self.video_path = video_path
        self.audio_path = audio_path
        self.max_length = max_length
        self.normalize = normalize
        self.cache_size = cache_size
        self.cache = {}

        if self.normalize:
            try:
                self.mean_audio = normalization_params["mean_audio"]
                self.std_audio = normalization_params["std_audio"]
                self.mean_video = normalization_params["mean_video"]
                self.std_video = normalization_params["std_video"]
            except KeyError as e:
                logging.error(f"Normalization parameter missing: {e}")
                raise

    def load_sample(self, filename):
        audio_csv = os.path.join(self.audio_path, f"{filename}.csv")
        video_parquet = os.path.join(self.video_path, f"{filename}.parquet")

        print(os.path.exists(audio_csv))
        print(os.path.exists(video_parquet))
        print(os.getcwd())
        print(audio_csv)
        print(video_parquet)

        if not os.path.exists(audio_csv):
            raise FileNotFoundError(f"Audio file not found: {audio_csv}")
        if not os.path.exists(video_parquet):
            raise FileNotFoundError(f"Video file not found: {video_parquet}")

        try:
            audio_data = pd.read_csv(audio_csv)
        except Exception as e:
            raise ValueError(f"Failed to read audio CSV for {filename}: {e}")

        try:
            video_data = pd.read_parquet(video_parquet)
        except Exception as e:
            raise ValueError(f"Failed to read video parquet for {filename}: {e}")

        # Group by keys
        audio_groups = audio_data.groupby(["video_id", "video_number"])
        video_groups = video_data.groupby(["video_id", "clip_num"])

        # Iterate through audio groups and find matching video groups
        for key in audio_groups.groups.keys():
            if key in video_groups.groups:
                audio_group = audio_groups.get_group(key)
                video_group = video_groups.get_group(key)

                try:
                    video_features = (
                        video_group.drop(
                            columns=["video_id", "clip_num", "desync", "frame_number"]
                        )
                        .astype(float)
                        .values
                    )
                except KeyError as e:
                    raise ValueError(
                        f"Missing video feature columns in {filename}: {e}"
                    )
                except Exception as e:
                    raise ValueError(
                        f"Error processing video features for {filename}: {e}"
                    )

                try:
                    audio_features = (
                        audio_group.drop(
                            columns=["video_id", "video_number", "frame_number"]
                        )
                        .astype(float)
                        .values
                    )
                except KeyError as e:
                    raise ValueError(
                        f"Missing audio feature columns in {filename}: {e}"
                    )
                except Exception as e:
                    raise ValueError(
                        f"Error processing audio features for {filename}: {e}"
                    )

                # Adjusted dimension check for audio features
                if audio_features.shape[1] != 118:
                    raise ValueError(
                        f"Unexpected audio feature dimension in {filename}: expected 118, got {audio_features.shape[1]}"
                    )

                min_length = min(len(audio_features), len(video_features))
                audio_features = audio_features[:min_length]
                video_features = video_features[:min_length]

                desync_values = video_group["desync"].astype(float).values[:min_length]
                if not np.all(desync_values == desync_values[0]):
                    raise ValueError(f"Inconsistent desync values in {filename}")

                y_offset = desync_values[0]

                if y_offset < -OFFSET or y_offset > OFFSET:
                    raise ValueError(f"y_offset {y_offset} out of range for {filename}")

                # Convert to tensors
                video_features = torch.tensor(video_features, dtype=torch.float)
                audio_features = torch.tensor(audio_features, dtype=torch.float)

                # Normalize if required
                if self.normalize:
                    video_features = (video_features - self.mean_video) / (
                        self.std_video + 1e-8
                    )
                    audio_features = (audio_features - self.mean_audio) / (
                        self.std_audio + 1e-8
                    )

                # Pad audio features from 118 to 120 dimensions
                if audio_features.shape[1] == 118:
                    padding = torch.zeros(
                        (audio_features.size(0), 2), dtype=torch.float
                    )
                    audio_features = torch.cat((audio_features, padding), dim=1)
                else:
                    raise ValueError(
                        f"Audio feature dimension after processing is not 118 for {filename}"
                    )

                # Pad or truncate video and audio features to max_length
                video_features = self.pad_or_truncate(video_features, self.max_length)
                audio_features = self.pad_or_truncate(audio_features, self.max_length)

                y_offset = torch.tensor(y_offset, dtype=torch.float)

                return video_features, audio_features, min_length, y_offset

    def pad_or_truncate(self, tensor, max_length):
        if tensor.size(0) > max_length:
            return tensor[:max_length]
        elif tensor.size(0) < max_length:
            padding_size = max_length - tensor.size(0)
            padding = torch.zeros((padding_size, tensor.size(1)), dtype=tensor.dtype)
            return torch.cat((tensor, padding), dim=0)
        return tensor

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        if filename in self.cache:
            return self.cache[filename]
        else:
            try:
                sample = self.load_sample(filename)
                if len(self.cache) >= self.cache_size:
                    remove_key = random.choice(list(self.cache.keys()))
                    del self.cache[remove_key]
                self.cache[filename] = sample
                return sample
            except FileNotFoundError as e:
                logging.error(f"File not found for sample {filename}: {e}")
            except ValueError as e:
                logging.error(f"Value error for sample {filename}: {e}")
            except Exception as e:
                logging.error(f"Unexpected error for sample {filename}: {e}")

            # Instead of returning None, raise an exception to prevent DataLoader from receiving None
            raise RuntimeError(f"Failed to load sample {filename}")


def load_and_intersect_data(video_dir, audio_dir):
    try:
        parquet_filenames = {
            os.path.splitext(f)[0]
            for f in os.listdir(video_dir)
            if f.endswith(".parquet")
        }
    except Exception as e:
        logging.error(f"Error listing video directory {video_dir}: {e}")
        raise

    try:
        csv_filenames = {
            os.path.splitext(f)[0]
            for f in os.listdir(audio_dir)
            if f.endswith(".csv") and os.path.getsize(os.path.join(audio_dir, f)) > 1922
        }
    except Exception as e:
        logging.error(f"Error listing audio directory {audio_dir}: {e}")
        raise

    intersecting_filenames = list(parquet_filenames.intersection(csv_filenames))
    logging.info(f"Number of intersecting filenames: {len(intersecting_filenames)}")
    return intersecting_filenames


def split_data(data, train_size=0.7, val_size=0.15):
    try:
        train_data, temp_data = train_test_split(
            data, train_size=train_size, random_state=42
        )
        val_data, test_data = train_test_split(
            temp_data, test_size=(1 - val_size / (1 - train_size)), random_state=42
        )
        return train_data, val_data, test_data
    except Exception as e:
        logging.error(f"Error splitting data: {e}")
        raise


def load_dataset(
    video_path,
    audio_path,
    normalization_params_path,
    max_data=None,
    save=False,
    batch_size=32,
):
    if max_data is None:
        max_data = {"train": None, "val": None, "test": None}

    combined_data = load_and_intersect_data(video_path, audio_path)
    train_filenames, val_filenames, test_filenames = split_data(combined_data)

    print(f"Total amount of data: {len(combined_data)}")

    # Sample the data if max_data limits are set
    if max_data.get("train") and len(train_filenames) > max_data["train"]:
        train_filenames = random.sample(train_filenames, max_data["train"])
    if max_data.get("val") and len(val_filenames) > max_data["val"]:
        val_filenames = random.sample(val_filenames, max_data["val"])
    if max_data.get("test") and len(test_filenames) > max_data["test"]:
        test_filenames = random.sample(test_filenames, max_data["test"])

    # Save filenames if required
    if save:
        for split, filenames in zip(
            ["train", "val", "test"], [train_filenames, val_filenames, test_filenames]
        ):
            try:
                with open(f"{split}_filenames.txt", "w") as f:
                    for filename in filenames:
                        f.write(f"{filename}\n")
                logging.info(f"Saved {split} filenames to {split}_filenames.txt")
            except Exception as e:
                logging.error(f"Failed to save {split} filenames: {e}")
                raise

    # Load normalization parameters
    if not os.path.exists(normalization_params_path):
        logging.error(
            f"Normalization parameters not found at {normalization_params_path}"
        )
        raise FileNotFoundError(
            f"Normalization parameters not found at {normalization_params_path}"
        )

    try:
        normalization_params = torch.load(normalization_params_path, map_location="cpu")
    except Exception as e:
        logging.error(f"Failed to load normalization parameters: {e}")
        raise

    # Create datasets
    try:
        train_dataset = VideoAudioDataset(
            train_filenames, video_path, audio_path, normalization_params
        )
        val_dataset = VideoAudioDataset(
            val_filenames, video_path, audio_path, normalization_params
        )
        test_dataset = VideoAudioDataset(
            test_filenames, video_path, audio_path, normalization_params
        )
    except Exception as e:
        logging.error(f"Failed to create datasets: {e}")
        raise

    logging.info(f"Train dataset size: {len(train_dataset)}")
    logging.info(f"Validation dataset size: {len(val_dataset)}")
    logging.info(f"Test dataset size: {len(test_dataset)}")

    # Create DataLoaders
    num_workers = min(4, os.cpu_count())
    try:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    except Exception as e:
        logging.error(f"Failed to create DataLoaders: {e}")
        raise

    return train_loader, val_loader, test_loader
