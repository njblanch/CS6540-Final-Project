# data_loading.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np

OFFSET = 15
MFCC = False


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
        self.video_path = video_path
        self.audio_path = audio_path
        self.max_length = max_length
        self.normalize = normalize
        self.cache_size = cache_size
        self.cache = {}

        if self.normalize:
            self.mean_audio = normalization_params["mean_audio"][
                :-2
            ]  # Change if normalization changes
            self.std_audio = normalization_params["std_audio"][:-2]
            if MFCC:
                self.mean_audio = self.mean_audio[1:13]
                self.std_audio = self.std_audio[1:13]
            self.mean_video = normalization_params["mean_video"]
            self.std_video = normalization_params["std_video"]
        print(self.mean_audio.shape)

        # This is here due to prior errors with loading the dataset
        self.filenames = []
        self.data = []
        for file in filenames:
            temp_data = self.load_sample(file)
            if temp_data is not None or temp_data:
                self.filenames.append(file)
                self.data.extend(temp_data)

    def load_sample(self, filename):
        audio_csv = os.path.join(self.audio_path, f"{filename}.csv")
        video_parquet = os.path.join(self.video_path, f"{filename}.parquet")

        if not (os.path.exists(audio_csv) and os.path.exists(video_parquet)):
            raise FileNotFoundError(f"Missing files for {filename}")
        temp_data = []
        try:
            audio_data = pd.read_csv(audio_csv)
            video_data = pd.read_parquet(video_parquet)

            audio_groups = audio_data.groupby(["video_id", "video_number"])
            video_groups = video_data.groupby(["video_id", "clip_num"])
            # print(audio_groups.groups.keys())
            # print(video_groups.groups.keys())
            for (video_id, clip_number), audio_group in audio_groups:
                # Ensure is in video groups too
                # print((video_id, clip_number))
                # print(list(video_groups.groups.keys()), flush=True)
                # print((f"{video_id}", f"{clip_number}") in list(video_groups.groups.keys()))
                if (f"{video_id}", f"{clip_number}") in list(
                    video_groups.groups.keys()
                ):
                    video_group = video_groups.get_group(
                        (f"{video_id}", f"{clip_number}")
                    )

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
                    # print(audio_features.shape)

                    if audio_features.shape[1] != 118:
                        # raise ValueError(
                        #     f"Unexpected audio feature dimension in {filename}"
                        # )
                        continue

                    # Only take MFCC data
                    if MFCC:
                        audio_features = audio_features[:, 1:13]

                    min_length = min(len(audio_features), len(video_features))
                    audio_features = audio_features[:min_length]
                    video_features = video_features[:min_length]

                    desync_values = (
                        video_group["desync"].astype(float).values[:min_length]
                    )
                    if not np.all(desync_values == desync_values[0]):
                        # This is really only useful in debugging
                        # raise ValueError(f"Inconsistent desync values in {filename}")
                        return None

                    y_offset = desync_values[0]

                    if y_offset < -OFFSET or y_offset > OFFSET:
                        # raise ValueError(f"y_offset {y_offset} out of range for {filename}")
                        return None

                    video_features = torch.tensor(video_features, dtype=torch.float)
                    audio_features = torch.tensor(audio_features, dtype=torch.float)

                    if self.normalize:
                        video_features = (video_features - self.mean_video) / (
                            self.std_video + 1e-8
                        )
                        audio_features = (audio_features - self.mean_audio) / (
                            self.std_audio + 1e-8
                        )

                    video_features = self.pad_or_truncate(
                        video_features, self.max_length
                    )
                    audio_features = self.pad_or_truncate(
                        audio_features, self.max_length
                    )

                    y_offset = torch.tensor(y_offset, dtype=torch.float)

                    # NOTE: Gian's version had a return here. This causes only a single data sequence per video. Fix that if
                    # you want to keep the loading during runtime thing
                    temp_data.append(
                        (video_features, audio_features, min_length, y_offset)
                    )

            return temp_data

        except Exception as e:
            print(f"Error loading sample {filename}: {e}", flush=True)
            return None
            # raise e

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
        data = self.data[idx]
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
            except Exception as e:
                print(f"Failed to load sample {filename}: {e}", flush=True)
                raise IndexError


def load_and_intersect_data(video_dir, audio_dir):
    parquet_filenames = {
        os.path.splitext(f)[0] for f in os.listdir(video_dir) if f.endswith(".parquet")
    }
    csv_filenames = {
        os.path.splitext(f)[0]
        for f in os.listdir(audio_dir)
        if f.endswith(".csv") and os.path.getsize(os.path.join(audio_dir, f)) > 1922
    }
    intersecting_filenames = list(parquet_filenames.intersection(csv_filenames))
    print(
        f"Number of intersecting filenames: {len(intersecting_filenames)}", flush=True
    )
    return intersecting_filenames


def split_data(data, train_size=0.7, val_size=0.15):
    train_data, temp_data = train_test_split(
        data, train_size=train_size, random_state=42
    )
    val_data, test_data = train_test_split(
        temp_data, test_size=(1 - val_size / (1 - train_size)), random_state=42
    )
    return train_data, val_data, test_data


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

    if max_data["train"] and len(train_filenames) > max_data["train"]:
        train_filenames = random.sample(train_filenames, max_data["train"])
    if max_data["val"] and len(val_filenames) > max_data["val"]:
        val_filenames = random.sample(val_filenames, max_data["val"])
    if max_data["test"] and len(test_filenames) > max_data["test"]:
        test_filenames = random.sample(test_filenames, max_data["test"])

    if save:
        for split, filenames in zip(
            ["train", "val", "test"], [train_filenames, val_filenames, test_filenames]
        ):
            with open(f"{split}_filenames.txt", "w") as f:
                for filename in filenames:
                    f.write(f"{filename}\n")
        print("Filenames saved.", flush=True)

    if not os.path.exists(normalization_params_path):
        raise FileNotFoundError(
            f"Normalization parameters not found at {normalization_params_path}"
        )

    normalization_params = torch.load(normalization_params_path)

    train_dataset = VideoAudioDataset(
        train_filenames, video_path, audio_path, normalization_params
    )
    val_dataset = VideoAudioDataset(
        val_filenames, video_path, audio_path, normalization_params
    )
    test_dataset = VideoAudioDataset(
        test_filenames, video_path, audio_path, normalization_params
    )

    print(f"Train dataset size: {len(train_dataset)}", flush=True)
    print(f"Validation dataset size: {len(val_dataset)}", flush=True)
    print(f"Test dataset size: {len(test_dataset)}", flush=True)

    num_workers = min(4, os.cpu_count())
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

    return train_loader, val_loader, test_loader
