import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import random
import torch.nn.functional as F
import numpy as np


class VideoAudioDataset(Dataset):
    # This provides all features per video_id and clip number - will have to change data loading things
    def __init__(self, filenames, video_path, audio_path, max_length=225):
        self.audio_features_list = []
        self.video_features_list = []
        self.y_offset_list = []
        self.video_id_list = []
        self.clip_number_list = []
        self.max_length = max_length

        for filename in filenames:
            audio_csv = os.path.join(audio_path, filename + '.csv')  # Assuming CSV for audio
            video_parquet = os.path.join(video_path, filename + '.parquet')  # Assuming parquet for video
            self.process_data(audio_csv, video_parquet)

        self.mean_audio = torch.mean(torch.cat(self.audio_features_list), dim=0)
        self.std_audio = torch.std(torch.cat(self.audio_features_list), dim=0)
        self.mean_video = torch.mean(torch.cat(self.video_features_list), dim=0)
        self.std_video = torch.std(torch.cat(self.video_features_list), dim=0)

        # Assuming audio_features_list and video_features_list contain all tensors
        all_audio_features = torch.cat(self.audio_features_list)
        all_video_features = torch.cat(self.video_features_list)

        # Compute global min and max for audio and video features
        self.audio_min = all_audio_features.min(dim=0).values
        self.audio_max = all_audio_features.max(dim=0).values
        self.video_min = all_video_features.min(dim=0).values
        self.video_max = all_video_features.max(dim=0).values

    def process_data(self, audio_csv, video_csv):
        if os.path.exists(audio_csv) and os.path.exists(video_csv):
            audio_data = pd.read_csv(audio_csv)
            video_data = pd.read_parquet(video_csv)
            # print(video_data.head())

            # Group using video_id and clip_number
            audio_groups = audio_data.groupby(['video_id', 'video_number'])
            video_groups = video_data.groupby(['video_id', 'clip_num'])

            # Process audio features
            for (video_id, clip_number), audio_group in audio_groups:
                # Ensure is in video groups too
                # print((video_id, clip_number))
                # print(list(video_groups.groups.keys()), flush=True)
                # print((f"{video_id}", f"{clip_number}") in list(video_groups.groups.keys()))
                if (f"{video_id}", f"{clip_number}") in list(video_groups.groups.keys()):
                    video_group = video_groups.get_group((f"{video_id}", f"{clip_number}"))
                    audio_features = torch.tensor(
                        audio_group.drop(
                            columns=['video_id', 'video_number', 'frame_number']).astype(float).values)  # All audio features

                    video_features = torch.tensor(video_group.drop(
                        columns=['video_id', 'clip_num', 'frame_number', 'desync']).astype(float).values)  # All video features

                    if audio_features.dtype != torch.float32:
                        audio_features = audio_features.float()

                    if video_features.dtype != torch.float32:
                        video_features = video_features.float()

                    # Ensure the number of rows match by trimming or padding
                    min_length = min(len(audio_features), len(video_features))
                    audio_features = audio_features[:min_length]  # Trim if needed
                    video_features = video_features[:min_length]  # Trim if needed

                    # Ensure the number of rows match
                    if len(audio_features) == len(video_features):
                        # Get most often occurring thing
                        y_offset_tensor = torch.tensor(video_group['desync'].astype(float).values[:min_length])  # Target variable
                        mode_value, _ = torch.mode(y_offset_tensor)
                        y_offset = mode_value

                        if y_offset < -7 or y_offset > 7:
                            continue

                        # Append lists
                        self.audio_features_list.append(audio_features)
                        self.video_features_list.append(video_features)
                        self.y_offset_list.append(y_offset)
                        self.video_id_list.append(video_id)
                        self.clip_number_list.append(clip_number)

    def __len__(self):
        return len(self.audio_features_list)

    def __getitem__(self, idx):
        video_id = self.video_id_list[idx]
        clip_number = self.clip_number_list[idx]

        y_offset = self.y_offset_list[idx]  # Target variable

        # Append features together
        audio_features = self.audio_features_list[idx] if idx < len(self.audio_features_list) else None
        video_features = self.video_features_list[idx] if idx < len(self.video_features_list) else None

        # audio_features = (audio_features - self.mean_audio) / (self.std_audio + 1e-8)
        # video_features = (video_features - self.mean_video) / (self.std_video + 1e-8)

        audio_features = min_max_normalize_global(audio_features, self.audio_min, self.audio_max)
        video_features = min_max_normalize_global(video_features, self.video_min, self.video_max)

        # Pad audio and video
        padded_audio = self.pad_or_truncate(audio_features, self.max_length).float()
        padded_video = self.pad_or_truncate(video_features, self.max_length).float()

        combined_features = torch.cat((padded_audio, padded_video), dim=-1)

        # attention_mask = self.create_attention_mask(len(combined_features), self.max_length)

        attention_mask = self.create_attention_mask(self.max_length, self.max_length)

        # y_offset = self.y_offset_list[idx] if idx < len(self.y_offset_list) else None
        #
        # padded_y_offset = self.pad_or_truncate(y_offset, self.max_length)

        return combined_features, attention_mask, y_offset

    def pad_or_truncate(self, tensor, max_length):
        # If the tensor has more than max_length elements along dimension 0, truncate it
        if tensor.size(0) > max_length:
            return tensor[:max_length]
        # If the tensor has fewer than max_length elements, pad with zeros
        if len(tensor.shape) == 1:
            padding = (0, max_length - tensor.size(0))
        else:
            padding = (0, 0, 0, max_length - tensor.size(0))  # (left, right, top, bottom)
        return F.pad(tensor, padding, value=0)

    def create_attention_mask(self, length, max_length):
        # Create a mask of True for actual data and False for padding
        mask = torch.ones(length, dtype=torch.bool)
        if length < max_length:
            pad = torch.zeros(max_length - length, dtype=torch.bool)
            mask = torch.cat((mask, pad))
        return mask


class VideoAudioDataset_OHE(Dataset):
    # This provides all features per video_id and clip number - will have to change data loading things
    def __init__(self, filenames, video_path, audio_path, max_length=225):
        self.audio_features_list = []
        self.video_features_list = []
        self.y_offset_list = []
        self.video_id_list = []
        self.clip_number_list = []
        self.max_length = max_length

        for filename in filenames:
            audio_csv = os.path.join(audio_path, filename + '.csv')  # Assuming CSV for audio
            video_parquet = os.path.join(video_path, filename + '.parquet')  # Assuming parquet for video
            self.process_data(audio_csv, video_parquet)

        self.mean_audio = torch.mean(torch.cat(self.audio_features_list), dim=0)
        self.std_audio = torch.std(torch.cat(self.audio_features_list), dim=0)
        self.mean_video = torch.mean(torch.cat(self.video_features_list), dim=0)
        self.std_video = torch.std(torch.cat(self.video_features_list), dim=0)

        # Assuming audio_features_list and video_features_list contain all tensors
        all_audio_features = torch.cat(self.audio_features_list)
        all_video_features = torch.cat(self.video_features_list)

        # Compute global min and max for audio and video features
        self.audio_min = all_audio_features.min(dim=0).values
        self.audio_max = all_audio_features.max(dim=0).values
        self.video_min = all_video_features.min(dim=0).values
        self.video_max = all_video_features.max(dim=0).values

        self.y_offset_list = one_hot_encode(self.y_offset_list)

    def process_data(self, audio_csv, video_csv):
        if os.path.exists(audio_csv) and os.path.exists(video_csv):
            audio_data = pd.read_csv(audio_csv)
            video_data = pd.read_parquet(video_csv)
            # print(video_data.head())

            # Group using video_id and clip_number
            audio_groups = audio_data.groupby(['video_id', 'video_number'])
            video_groups = video_data.groupby(['video_id', 'clip_num'])

            # Process audio features
            for (video_id, clip_number), audio_group in audio_groups:
                # Ensure is in video groups too
                # print((video_id, clip_number))
                # print(list(video_groups.groups.keys()), flush=True)
                # print((f"{video_id}", f"{clip_number}") in list(video_groups.groups.keys()))
                if (f"{video_id}", f"{clip_number}") in list(video_groups.groups.keys()):
                    video_group = video_groups.get_group((f"{video_id}", f"{clip_number}"))
                    audio_features = torch.tensor(
                        audio_group.drop(
                            columns=['video_id', 'video_number', 'frame_number']).astype(float).values)  # All audio features

                    video_features = torch.tensor(video_group.drop(
                        columns=['video_id', 'clip_num', 'frame_number', 'desync']).astype(float).values)  # All video features

                    if audio_features.dtype != torch.float32:
                        audio_features = audio_features.float()

                    if video_features.dtype != torch.float32:
                        video_features = video_features.float()

                    # Ensure the number of rows match by trimming or padding
                    min_length = min(len(audio_features), len(video_features))
                    audio_features = audio_features[:min_length]  # Trim if needed
                    video_features = video_features[:min_length]  # Trim if needed

                    # Ensure the number of rows match
                    if len(audio_features) == len(video_features):
                        # Get most often occurring thing
                        y_offset_tensor = torch.tensor(video_group['desync'].astype(float).values[:min_length])  # Target variable
                        mode_value, _ = torch.mode(y_offset_tensor)
                        y_offset = mode_value

                        if y_offset < -7 or y_offset > 7:
                            continue

                        # Append lists
                        self.audio_features_list.append(audio_features)
                        self.video_features_list.append(video_features)
                        self.y_offset_list.append(y_offset)
                        self.video_id_list.append(video_id)
                        self.clip_number_list.append(clip_number)

    def __len__(self):
        return len(self.audio_features_list)

    def __getitem__(self, idx):
        video_id = self.video_id_list[idx]
        clip_number = self.clip_number_list[idx]

        y_offset = self.y_offset_list[idx]  # Target variable

        # Append features together
        audio_features = self.audio_features_list[idx] if idx < len(self.audio_features_list) else None
        video_features = self.video_features_list[idx] if idx < len(self.video_features_list) else None

        # audio_features = (audio_features - self.mean_audio) / (self.std_audio + 1e-8)
        # video_features = (video_features - self.mean_video) / (self.std_video + 1e-8)

        audio_features = min_max_normalize_global(audio_features, self.audio_min, self.audio_max)
        video_features = min_max_normalize_global(video_features, self.video_min, self.video_max)

        # Pad audio and video
        padded_audio = self.pad_or_truncate(audio_features, self.max_length).float()
        padded_video = self.pad_or_truncate(video_features, self.max_length).float()

        combined_features = torch.cat((padded_audio, padded_video), dim=-1)

        # attention_mask = self.create_attention_mask(len(combined_features), self.max_length)

        attention_mask = self.create_attention_mask(self.max_length, self.max_length)

        # y_offset = self.y_offset_list[idx] if idx < len(self.y_offset_list) else None
        #
        # padded_y_offset = self.pad_or_truncate(y_offset, self.max_length)

        return combined_features, attention_mask, y_offset

    def pad_or_truncate(self, tensor, max_length):
        # If the tensor has more than max_length elements along dimension 0, truncate it
        if tensor.size(0) > max_length:
            return tensor[:max_length]
        # If the tensor has fewer than max_length elements, pad with zeros
        if len(tensor.shape) == 1:
            padding = (0, max_length - tensor.size(0))
        else:
            padding = (0, 0, 0, max_length - tensor.size(0))  # (left, right, top, bottom)
        return F.pad(tensor, padding, value=0)

    def create_attention_mask(self, length, max_length):
        # Create a mask of True for actual data and False for padding
        mask = torch.ones(length, dtype=torch.bool)
        if length < max_length:
            pad = torch.zeros(max_length - length, dtype=torch.bool)
            mask = torch.cat((mask, pad))
        return mask


def one_hot_encode(labels):
    # Define classes from -7 to 7
    class_labels = list(range(-7, 8))  # This creates the list: [-7, -6, ..., 0, ..., 6, 7]
    unique_classes = np.array(class_labels)  # Ensure it's a NumPy array

    # Create a mapping from class label to index
    class_to_index = {cls: idx for idx, cls in enumerate(unique_classes)}

    # Initialize the one-hot matrix
    one_hot_matrix = np.zeros((len(labels), len(unique_classes)), dtype=int)

    # Fill the one-hot matrix
    for idx, label in enumerate(labels):
        label = int(label)
        print(label)
        if label in class_to_index:  # Ensure the label is within the defined classes
            one_hot_matrix[idx, class_to_index[label]] = 1
        else:
            raise ValueError(f"Label {label} is not in the range of -7 to 7.")
    print(one_hot_matrix, flush=True)
    return one_hot_matrix


def min_max_normalize_global(tensor, tensor_min, tensor_max):
    return 2 * (tensor - tensor_min) / (tensor_max - tensor_min + 1e-8) - 1


def normalize_tensor(tensor, mean, std):
    return (tensor - mean) / std


def load_and_intersect_data(dir1, dir2):
    # Gather all filenames from both directories
    parquet_filenames = set()
    csv_filenames = set()

    # Process first directory (parquet files)
    for filename in os.listdir(dir1):
        if filename.endswith('.parquet'):
            name_without_extension = os.path.splitext(os.path.basename(filename))[0]
            parquet_filenames.add(name_without_extension)

    # Process second directory (csv files)
    for filename in os.listdir(dir2):
        if filename.endswith('.csv'):
            file_path = os.path.join(dir2, filename)  # Full path for size check
            if os.path.getsize(file_path) > 1922:  # Ensure file has data
                name_without_extension = os.path.splitext(os.path.basename(filename))[0]
                csv_filenames.add(name_without_extension)

    # Find the intersection of filenames
    intersecting_filenames = list(parquet_filenames.intersection(csv_filenames))
    print(len(intersecting_filenames), flush=True)
    return intersecting_filenames

    # Gather all filenames from both directories
    all_filenames = []


def split_data(data, train_size=0.7, val_size=0.15):
    # Split the data into train and temp (validation + test)
    train_data, temp_data = train_test_split(data, train_size=train_size, random_state=42)

    # Calculate the adjusted validation size from temp_data
    val_data, test_data = train_test_split(temp_data, test_size=(1 - val_size / (1 - train_size)), random_state=42)

    return train_data, val_data, test_data


def calculate_mean_std(dataset):
    """
    Calculate the mean and standard deviation of the dataset features.
    Assumes the dataset returns a tuple (video, audio) in __getitem__.
    """
    all_features = []
    all_masks = []

    for idx in range(len(dataset)):
        features_, mask_, _ = dataset[idx]
        all_features.append(features_)
        all_masks.append(mask_)

    # Stack all audio features and masks
    all_audio_features = torch.stack(all_features)  # Shape: (num_samples, sequence_length, num_features)
    all_masks = torch.stack(all_masks)  # Shape: (num_samples, sequence_length)

    # Apply the mask: Set padded values to zero and create a sum mask
    masked_features = all_audio_features * all_masks.unsqueeze(
        -1)  # Shape: (num_samples, sequence_length, num_features)

    # Count valid entries (1s in mask)
    num_valid = all_masks.sum(dim=1, keepdim=True)  # Shape: (num_samples, 1)

    # Calculate mean and std ignoring padding
    mean = masked_features.sum(dim=1) / num_valid  # Shape: (num_samples, num_features)
    mean = mean.sum(dim=0) / len(dataset)  # Final mean: Shape: (num_features,)

    # To calculate std, we need to sum the squared differences
    squared_diff = (masked_features - mean.unsqueeze(0).unsqueeze(1)) ** 2
    std = squared_diff.sum(dim=1) / num_valid  # Shape: (num_samples, num_features)
    std = std.sum(dim=0) / len(dataset)

    return mean, std


def load_dataset(video_path, audio_path, max_data=None, save=False, batch_size=32):
    # Load and union data
    skip_downsample = False
    if max_data is None:
        max_data = {"train": 5000, "test": 1000, "val": 1000}
    elif max_data["train"] == -1:
        skip_downsample = True

    combined_data = load_and_intersect_data(video_path, audio_path)

    # Split the data
    train_filenames, val_filenames, test_filenames = split_data(combined_data)

    # limit dataset sizes
    if not skip_downsample:
        print(f"Downsampling: {max_data}", flush=True)
        if len(train_filenames) > max_data["train"]:
            train_filenames = random.sample(train_filenames, max_data["train"])

        if len(val_filenames) > max_data["val"]:
            val_filenames = random.sample(val_filenames, max_data["val"])

        if len(test_filenames) > max_data["test"]:
            test_filenames = random.sample(test_filenames, max_data["test"])

    # save the filenames
    if save:
        print("Saving files")
        with open('train_filenames.txt', 'w') as train_file:
            for filename in train_filenames:
                train_file.write(f"{filename}\n")

        with open('val_filenames.txt', 'w') as val_file:
            for filename in val_filenames:
                val_file.write(f"{filename}\n")

        with open('test_filenames.txt', 'w') as test_file:
            for filename in test_filenames:
                test_file.write(f"{filename}\n")

    # Create DataLoaders for each split (assuming the dataset class handles loading from the filenames)
    train_dataset = VideoAudioDataset(train_filenames, video_path, audio_path)
    val_dataset = VideoAudioDataset(val_filenames, video_path, audio_path)
    test_dataset = VideoAudioDataset(test_filenames, video_path, audio_path)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def load_dataset_OHE(video_path, audio_path, max_data=None, save=False, batch_size=32):
    # Load and union data
    skip_downsample = False
    if max_data is None:
        max_data = {"train": 5000, "test": 1000, "val": 1000}
    elif max_data["train"] == -1:
        skip_downsample = True

    combined_data = load_and_intersect_data(video_path, audio_path)

    # Split the data
    train_filenames, val_filenames, test_filenames = split_data(combined_data)

    # limit dataset sizes
    if not skip_downsample:
        print(f"Downsampling: {max_data}", flush=True)
        if len(train_filenames) > max_data["train"]:
            train_filenames = random.sample(train_filenames, max_data["train"])

        if len(val_filenames) > max_data["val"]:
            val_filenames = random.sample(val_filenames, max_data["val"])

        if len(test_filenames) > max_data["test"]:
            test_filenames = random.sample(test_filenames, max_data["test"])

    # save the filenames
    if save:
        print("Saving files")
        with open('train_filenames_ohe.txt', 'w') as train_file:
            for filename in train_filenames:
                train_file.write(f"{filename}\n")

        with open('val_filenames_ohe.txt', 'w') as val_file:
            for filename in val_filenames:
                val_file.write(f"{filename}\n")

        with open('test_filenames_ohe.txt', 'w') as test_file:
            for filename in test_filenames:
                test_file.write(f"{filename}\n")

    # Create DataLoaders for each split (assuming the dataset class handles loading from the filenames)
    train_dataset = VideoAudioDataset_OHE(train_filenames, video_path, audio_path)
    val_dataset = VideoAudioDataset_OHE(val_filenames, video_path, audio_path)
    test_dataset = VideoAudioDataset_OHE(test_filenames, video_path, audio_path)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Use with singular files
    train_audio_csv = 'path/to/train_audio.csv'
    train_video_csv = 'path/to/train_video.csv'
    val_audio_csv = 'path/to/val_audio.csv'
    val_video_csv = 'path/to/val_video.csv'
    test_audio_csv = 'path/to/test_audio.csv'
    test_video_csv = 'path/to/test_video.csv'

    # Create dataset instances
    train_dataset = VideoAudioDataset(train_audio_csv, train_video_csv)
    val_dataset = VideoAudioDataset(val_audio_csv, val_video_csv)
    test_dataset = VideoAudioDataset(test_audio_csv, test_video_csv)

    # Create DataLoader instances
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Example of accessing a single item from the dataset
    for video_id, clip_number, clip_type, features in train_loader:
        print(video_id, clip_number, clip_type, features.shape)  # Features will be a tensor
        break  # Only print the first batch
