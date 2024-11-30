# data_loading.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import logging
from collections import OrderedDict
from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

VIDEO_FEATURES = 1024
AUDIO_FEATURES = 120


class VideoAudioDataset(Dataset):
    def __init__(
        self,
        clip_list,
        video_path,
        audio_path,
        normalization_params,
        max_length=225,
        normalize=True,
        cache_size=10000,
        # y_mean=None,
        # y_std=None,
    ):
        """
        Args:
            clip_list (list of tuples): Each tuple contains (filename, clip_num).
            video_path (str): Path to the video parquet files.
            audio_path (str): Path to the audio CSV files.
            normalization_params (dict): Normalization parameters.
            max_length (int): Maximum sequence length.
            normalize (bool): Whether to apply normalization.
            cache_size (int): Number of files to cache in memory.
        """
        self.clip_list = clip_list
        self.video_path = video_path
        self.audio_path = audio_path
        self.normalization_params = normalization_params
        self.max_length = max_length
        self.normalize = normalize
        self.cache_size = cache_size
        self.cache = OrderedDict()  # To maintain insertion order for LRU

        # self.y_mean = y_mean
        # self.y_std = y_std

        if self.normalize:
            # Correct tensor copying to avoid warnings
            self.mean_video = (
                self.normalization_params["mean_video"]
                .clone()
                .detach()
                .float()
                .unsqueeze(0)
            )
            self.std_video = (
                self.normalization_params["std_video"]
                .clone()
                .detach()
                .float()
                .unsqueeze(0)
            )
            self.mean_audio = (
                self.normalization_params["mean_audio"]
                .clone()
                .detach()
                .float()
                .unsqueeze(0)
            )
            self.std_audio = (
                self.normalization_params["std_audio"]
                .clone()
                .detach()
                .float()
                .unsqueeze(0)
            )

    def __len__(self):
        return len(self.clip_list)

    def __getitem__(self, idx):
        filename, clip_num = self.clip_list[idx]

        if filename in self.cache and clip_num in self.cache[filename]:
            return self.cache[filename][clip_num]

        if filename not in self.cache:
            try:
                video_file = os.path.join(self.video_path, f"{filename}.parquet")
                audio_file = os.path.join(self.audio_path, f"{filename}.csv")

                video_df = pd.read_parquet(video_file)
                video_df["clip_num"] = video_df["clip_num"].astype(int)
                unique_clips_video = set(video_df["clip_num"].unique())

                audio_df = pd.read_csv(audio_file)
                unique_clips_audio = set(audio_df["video_number"].unique())

                common_clips = unique_clips_video.intersection(unique_clips_audio)
                if not common_clips:
                    logging.warning(
                        f"No common clips found for file {filename}. Skipping."
                    )
                    self.cache[filename] = {}
                else:
                    processed_clips = {}
                    for clip in common_clips:
                        # Process video data
                        video_clip_df = video_df[video_df["clip_num"] == clip]
                        if video_clip_df.empty:
                            logging.warning(
                                f"No video data for clip_num {clip} in file {filename}. Skipping clip."
                            )
                            continue

                        y_offset = video_clip_df["desync"].values.astype(float)
                        y_offset = y_offset.mean()

                        # if self.y_mean is not None and self.y_std is not None:
                        #     y_offset = (y_offset - self.y_mean) / self.y_std

                        video_features = video_clip_df.drop(
                            columns=["video_id", "clip_num", "desync", "frame_number"]
                        )

                        video_features = video_features.apply(
                            pd.to_numeric, errors="coerce"
                        )

                        video_features = video_features.fillna(0).values.astype(
                            np.float32
                        )

                        audio_clip_df = audio_df[audio_df["video_number"] == clip]
                        if audio_clip_df.empty:
                            logging.warning(
                                f"No audio data for video_number {clip} in file {filename}. Skipping clip."
                            )
                            continue

                        audio_features = audio_clip_df.drop(
                            columns=["video_id", "video_number"]
                        )

                        audio_features = audio_features.apply(
                            pd.to_numeric, errors="coerce"
                        )

                        audio_features = audio_features.fillna(0).values.astype(
                            np.float32
                        )

                        # Zero pad the audio features to go from 118 to 120 features if necessary
                        if audio_features.shape[1] < 120:
                            audio_features = np.pad(
                                audio_features,
                                ((0, 0), (0, 120 - audio_features.shape[1])),
                                mode="constant",
                            )

                        video_features_tensor = torch.tensor(
                            video_features, dtype=torch.float
                        )
                        audio_features_tensor = torch.tensor(
                            audio_features, dtype=torch.float
                        )

                        # print("Mean Video Shape:", self.mean_video.shape)
                        # print("Std Video Shape:", self.std_video.shape)
                        # print("Mean Audio Shape:", self.mean_audio.shape)
                        # print("Std Audio Shape:", self.std_audio.shape)

                        # print("Video Features Shape:", video_features_tensor.shape)
                        # print("Audio Features Shape:", audio_features_tensor.shape)

                        # Normalize features
                        if self.normalize:
                            video_features_tensor = (
                                video_features_tensor - self.mean_video
                            ) / self.std_video
                            audio_features_tensor = (
                                audio_features_tensor - self.mean_audio
                            ) / self.std_audio

                        # Truncate sequences to max_length
                        seq_len = min(
                            video_features_tensor.size(0),
                            audio_features_tensor.size(0),
                            self.max_length,
                        )
                        video_features_tensor = video_features_tensor[:seq_len]
                        audio_features_tensor = audio_features_tensor[:seq_len]

                        if seq_len < self.max_length:
                            pad_len = self.max_length - seq_len
                            video_features_tensor = torch.cat(
                                (
                                    video_features_tensor,
                                    torch.zeros(pad_len, VIDEO_FEATURES),
                                )
                            )
                            audio_features_tensor = torch.cat(
                                (
                                    audio_features_tensor,
                                    torch.zeros(pad_len, AUDIO_FEATURES),
                                )
                            )

                        # Convert target to tensor
                        y_offset_tensor = torch.tensor(y_offset, dtype=torch.float)

                        # Store the processed clip
                        processed_clips[clip] = (
                            video_features_tensor,
                            audio_features_tensor,
                            seq_len,
                            y_offset_tensor,
                        )

                    # Add processed clips to cache
                    self.cache[filename] = processed_clips

                    # Implement LRU cache: remove oldest entry if cache exceeds size
                    if len(self.cache) > self.cache_size:
                        oldest_filename, _ = self.cache.popitem(last=False)
                        logging.info(
                            f"Cache size exceeded. Removed oldest cached file: {oldest_filename}"
                        )

            except Exception as e:
                logging.error(
                    f"Error processing file {filename}: {e}. Skipping all clips from this file."
                )
                self.cache[filename] = {}

        # Retrieve the specific clip from cache
        if clip_num not in self.cache[filename]:
            raise ValueError(
                f"Clip_num {clip_num} for file {filename} is not available in the cache."
            )

        return self.cache[filename][clip_num]


def load_dataset(
    video_path,
    audio_path,
    normalization_params_path,
    data_sizes={"train": 1000, "val": 200, "test": 200},
    save=False,
    load=False,
):
    """
    Loads the dataset, handling multiple clips per file and ensuring unique video IDs per split.

    Args:
        video_path (str): Path to the video parquet files.
        audio_path (str): Path to the audio CSV files.
        normalization_params_path (str): Path to normalization parameters.
        data_sizes (dict): Number of samples (clips) for train, val, test.
        save (bool): Whether to save the datasets.
        load (bool): Whether to load the datasets from saved files.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    # save and load can't both be True
    if save and load:
        raise ValueError("save and load can't both be True")
    # Ensuring data_sizes is a dictionary that looks like
    # {"train": int, "val": int, "test": int}
    try:
        assert isinstance(data_sizes, dict)
        assert "train" in data_sizes
        assert "val" in data_sizes
        assert "test" in data_sizes
    except AssertionError:
        raise ValueError(
            "data_sizes must be a dict with keys 'train', 'val', and 'test'"
        )

    if load:
        train_dataset = torch.load("dataset/train_dataset_100k.pt")
        val_dataset = torch.load("dataset/val_dataset_100k.pt")
        test_dataset = torch.load("dataset/test_dataset_100k.pt")
        return train_dataset, val_dataset, test_dataset

    normalization_params = torch.load(normalization_params_path)
    all_filenames = [
        f.split(".")[0] for f in os.listdir(video_path) if f.endswith(".parquet")
    ]

    logging.info(f"Found {len(all_filenames)} video files.")

    random.shuffle(all_filenames)
    num_train = data_sizes["train"]
    num_val = data_sizes["val"]
    num_test = data_sizes["test"]
    num_total = num_train + num_val + num_test

    train_clips = []
    val_clips = []
    test_clips = []

    train_count = 0
    val_count = 0
    test_count = 0

    with tqdm(total=num_total, desc="Loading dataset") as pbar:
        for filename in all_filenames:
            if (
                train_count >= num_train
                and val_count >= num_val
                and test_count >= num_test
            ):
                break

            video_file = os.path.join(video_path, f"{filename}.parquet")
            audio_file = os.path.join(audio_path, f"{filename}.csv")

            # Load video data to find clip_num
            try:
                video_df = pd.read_parquet(video_file)
            except Exception as e:
                logging.error(f"Error loading video file {video_file}: {e}")
                pbar.update(4)
                continue

            video_df["clip_num"] = video_df["clip_num"].astype(int)
            unique_clips_video = video_df["clip_num"].unique()

            # Load audio data to find video_number
            try:
                audio_df = pd.read_csv(audio_file)
            except Exception as e:
                logging.error(f"Error loading audio file {audio_file}: {e}")
                pbar.update(4)
                continue

            audio_df["video_number"] = audio_df["video_number"].astype(int)
            unique_clips_audio = audio_df["video_number"].unique()

            # Matching video and audio clips
            common_clips = set(unique_clips_video).intersection(set(unique_clips_audio))
            clips = list(common_clips)
            if not clips:
                logging.warning(f"No common clips found for file {filename}. Skipping.")
                logging.warning(f"Unique clips in video: {unique_clips_video}")
                logging.warning(f"Unique clips in audio: {unique_clips_audio}")
                pbar.update(4)
                continue

            # Assign the entire video to a split based on current counts
            if train_count + len(clips) <= num_train:
                for clip_num in clips:
                    train_clips.append((filename, clip_num))
                train_count += len(clips)
            elif val_count + len(clips) <= num_val:
                for clip_num in clips:
                    val_clips.append((filename, clip_num))
                val_count += len(clips)
            elif test_count + len(clips) <= num_test:
                for clip_num in clips:
                    test_clips.append((filename, clip_num))
                test_count += len(clips)
            else:
                # Assign to the split with the least current count
                min_split = min(
                    [("train", train_count), ("val", val_count), ("test", test_count)],
                    key=lambda x: x[1],
                )[0]
                if min_split == "train":
                    for clip_num in clips:
                        train_clips.append((filename, clip_num))
                    train_count += len(clips)
                elif min_split == "val":
                    for clip_num in clips:
                        val_clips.append((filename, clip_num))
                    val_count += len(clips)
                else:
                    for clip_num in clips:
                        test_clips.append((filename, clip_num))
                    test_count += len(clips)

            pbar.update(4)

    logging.info(
        f"Assigned clips\nTrain: {len(train_clips)} (Requested: {num_train}), "
        f"\nVal: {len(val_clips)} (Requested: {num_val}), "
        f"\nTest: {len(test_clips)} (Requested: {num_test})"
    )

    # Verify if requested counts are met
    if (
        len(train_clips) < num_train
        or len(val_clips) < num_val
        or len(test_clips) < num_test
    ):
        logging.warning(
            "Requested data sizes not fully met due to the distribution of clips per video. "
            "Consider adjusting data_sizes or ensuring more data."
        )

    # Create datasets
    train_dataset = VideoAudioDataset(
        train_clips, video_path, audio_path, normalization_params, max_length=225
    )
    val_dataset = VideoAudioDataset(
        val_clips, video_path, audio_path, normalization_params, max_length=225
    )
    test_dataset = VideoAudioDataset(
        test_clips, video_path, audio_path, normalization_params, max_length=225
    )

    # Save datasets
    if save:
        torch.save(train_dataset, "train_dataset.pt")
        torch.save(val_dataset, "val_dataset.pt")
        torch.save(test_dataset, "test_dataset.pt")
        logging.info("Datasets have been saved successfully.")

    return train_dataset, val_dataset, test_dataset
