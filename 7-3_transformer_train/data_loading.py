# data_loading.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from functools import lru_cache

OFFSET = 7


def one_hot_encode(labels, offset=OFFSET):
    class_labels = list(range(-offset, offset + 1))  # [-7, -6, ..., 0, ..., 6, 7]
    unique_classes = np.array(class_labels)
    class_to_index = {cls: idx for idx, cls in enumerate(unique_classes)}
    one_hot_matrix = np.zeros((len(labels), len(unique_classes)), dtype=int)
    for idx, label in enumerate(labels):
        label = int(label)
        if label in class_to_index:
            one_hot_matrix[idx, class_to_index[label]] = 1
        else:
            raise ValueError(
                f"Label {label} is not in the range of -{offset} to {offset}."
            )
    return one_hot_matrix


class VideoAudioDataset(Dataset):
    def __init__(
        self,
        filenames,
        video_path,
        audio_path,
        normalization_params=None,
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
            if normalization_params is None:
                raise ValueError(
                    "Normalization parameters must be provided if normalize=True."
                )
            self.mean_audio = normalization_params["mean_audio"]
            self.std_audio = normalization_params["std_audio"]
            self.mean_video = normalization_params["mean_video"]
            self.std_video = normalization_params["std_video"]

    @lru_cache(maxsize=1000)
    def load_sample(self, filename):
        """
        Load and preprocess a single data sample.
        Cached to avoid redundant processing.
        """
        audio_csv = os.path.join(self.audio_path, f"{filename}.csv")
        video_parquet = os.path.join(self.video_path, f"{filename}.parquet")

        if not (os.path.exists(audio_csv) and os.path.exists(video_parquet)):
            raise FileNotFoundError(f"Missing files for {filename}")

        try:
            audio_data = pd.read_csv(audio_csv)
            video_data = pd.read_parquet(video_parquet)

            # Group by video_id and clip_number
            audio_groups = audio_data.groupby(["video_id", "video_number"])
            video_groups = video_data.groupby(["video_id", "clip_num"])

            # Assuming one group per file
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
                if audio_features.shape[1] < 120:
                    padding_size = 120 - audio_features.shape[1]
                    audio_features = np.pad(
                        audio_features, ((0, 0), (0, padding_size)), "constant"
                    )
                elif audio_features.shape[1] > 120:
                    audio_features = audio_features[:, :120]

                # Ensure matching sequence lengths
                min_length = min(len(audio_features), len(video_features))
                audio_features = audio_features[:min_length]
                video_features = video_features[:min_length]

                # Compute target: mean of 'desync'
                y_offset = (
                    video_group["desync"].astype(float).values[:min_length].mean()
                )

                if y_offset < -OFFSET or y_offset > OFFSET:
                    raise ValueError(f"y_offset {y_offset} out of range for {filename}")

                # Convert to tensors
                video_features = torch.tensor(video_features, dtype=torch.float)
                audio_features = torch.tensor(audio_features, dtype=torch.float)

                # Normalize
                if self.normalize:
                    video_features = (video_features - self.mean_video) / (
                        self.std_video + 1e-8
                    )
                    audio_features = (audio_features - self.mean_audio) / (
                        self.std_audio + 1e-8
                    )

                # Pad or truncate to max_length
                video_features = self.pad_or_truncate(video_features, self.max_length)
                audio_features = self.pad_or_truncate(audio_features, self.max_length)

                # Convert target to tensor
                y_offset = torch.tensor(y_offset, dtype=torch.float)

                return video_features, audio_features, min_length, y_offset

        except Exception as e:
            print(f"Error loading sample {filename}: {e}", flush=True)
            raise e

    @staticmethod
    def pad_or_truncate(tensor, max_length):
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
                # Add to cache
                if len(self.cache) >= self.cache_size:
                    # Remove a random item from cache to make space
                    remove_key = random.choice(list(self.cache.keys()))
                    del self.cache[remove_key]
                self.cache[filename] = sample
                return sample
            except Exception as e:
                print(f"Failed to load sample {filename}: {e}", flush=True)
                # Optionally, implement logic to skip or substitute
                # For now, re-raise the exception
                raise e


def load_and_intersect_data(dir1, dir2):
    # Gather all filenames from both directories
    parquet_filenames = set()
    csv_filenames = set()

    # Process first directory (parquet files)
    for filename in os.listdir(dir1):
        if filename.endswith(".parquet"):
            name_without_extension = os.path.splitext(os.path.basename(filename))[0]
            parquet_filenames.add(name_without_extension)

    # Process second directory (csv files)
    for filename in os.listdir(dir2):
        if filename.endswith(".csv"):
            file_path = os.path.join(dir2, filename)  # Full path for size check
            if os.path.getsize(file_path) > 1922:  # Ensure file has data
                name_without_extension = os.path.splitext(os.path.basename(filename))[0]
                csv_filenames.add(name_without_extension)

    # Find the intersection of filenames
    intersecting_filenames = list(parquet_filenames.intersection(csv_filenames))
    print(
        f"Number of intersecting filenames: {len(intersecting_filenames)}", flush=True
    )
    return intersecting_filenames


def split_data(data, train_size=0.7, val_size=0.15):
    # Split the data into train and temp (validation + test)
    train_data, temp_data = train_test_split(
        data,
        train_size=train_size,
        random_state=42,
    )

    # Calculate the adjusted validation size from temp_data
    val_data, test_data = train_test_split(
        temp_data,
        test_size=(1 - val_size / (1 - train_size)),  # random_state=42
        random_state=42,
    )

    return train_data, val_data, test_data


def load_dataset(
    video_path,
    audio_path,
    normalization_params_path,
    max_data=None,
    save=False,
    batch_size=32,
    train_size=1000,
):
    if train_size:
        test_size = train_size // 5
        val_size = train_size // 5
    # Load and union data
    skip_downsample = False
    if max_data is None:
        skip_downsample = True

    combined_data = load_and_intersect_data(video_path, audio_path)

    # Split the data
    train_filenames, val_filenames, test_filenames = split_data(combined_data)

    # Limit dataset sizes
    if not skip_downsample:
        print(f"Downsampling: {max_data}", flush=True)
        if len(train_filenames) > max_data["train"]:
            train_filenames = random.sample(train_filenames, max_data["train"])

        if len(val_filenames) > max_data["val"]:
            val_filenames = random.sample(val_filenames, max_data["val"])

        if len(test_filenames) > max_data["test"]:
            test_filenames = random.sample(test_filenames, max_data["test"])

    # Save the filenames
    if save:
        print("Saving files", flush=True)
        with open("train_filenames.txt", "w") as train_file:
            for filename in train_filenames:
                train_file.write(f"{filename}\n")

        with open("val_filenames.txt", "w") as val_file:
            for filename in val_filenames:
                val_file.write(f"{filename}\n")

        with open("test_filenames.txt", "w") as test_file:
            for filename in test_filenames:
                test_file.write(f"{filename}\n")
        print("Files saved", flush=True)

    # Load normalization parameters
    if not os.path.exists(normalization_params_path):
        raise FileNotFoundError(
            f"Normalization parameters file not found at {normalization_params_path}"
        )

    normalization_params = torch.load(normalization_params_path)

    print("Loading datasets", flush=True)
    # Create DataLoaders for each split
    train_dataset = VideoAudioDataset(
        train_filenames,
        video_path,
        audio_path,
        normalization_params=normalization_params,
    )
    print("train_dataset loaded", flush=True)
    val_dataset = VideoAudioDataset(
        val_filenames, video_path, audio_path, normalization_params=normalization_params
    )
    print("val_dataset loaded", flush=True)
    test_dataset = VideoAudioDataset(
        test_filenames,
        video_path,
        audio_path,
        normalization_params=normalization_params,
    )
    print("test_dataset loaded", flush=True)

    print(f"Train dataset size: {len(train_dataset)}", flush=True)
    print(f"Validation dataset size: {len(val_dataset)}", flush=True)
    print(f"Test dataset size: {len(test_dataset)}", flush=True)

    num_workers = min(4, os.cpu_count())
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
