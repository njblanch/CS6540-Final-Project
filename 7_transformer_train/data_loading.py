import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader


class VideoAudioDataset(Dataset):
    # This provides all features per video_id and clip number - will have to change data loading things
    def __init__(self, filenames, video_path, audio_path):
        self.sub_clips = []

        for filename in filenames:
            audio_csv = os.path.join(audio_path, filename + '.csv')  # Assuming CSV for audio
            video_csv = os.path.join(video_path, filename + '.csv')  # Assuming CSV for video
            self.process_data(audio_csv, video_csv)

    def process_data(self, audio_csv, video_csv):
        if os.path.exists(audio_csv) and os.path.exists(video_csv):
            audio_data = pd.read_csv(audio_csv)
            video_data = pd.read_csv(video_csv)

            # Group using videoid (if for some reason multiple things together) and also clip number
            audio_groups = audio_data.groupby(['video_id', 'clip_number'])
            video_groups = video_data.groupby(['video_id', 'clip_number'])

            # Iterate through and load audio features #TODO: change to load all features
            for (video_id, clip_number), audio_group in audio_groups:
                audio_features = torch.tensor(audio_group['audio_features'].tolist())  # Adjust based on actual column
                self.sub_clips.append((video_id, clip_number, 'audio', audio_features))
            # TODO: change to load all features
            for (video_id, clip_number), video_group in video_groups:
                video_features = torch.tensor(video_group['video_features'].tolist())  # Adjust based on actual column
                self.sub_clips.append((video_id, clip_number, 'video', video_features))

    def __len__(self):
        return len(self.sub_clips)

    def __getitem__(self, idx):
        video_id, clip_number, clip_type, features = self.sub_clips[idx]
        return video_id, clip_number, clip_type, features


def load_and_union_data(dir1, dir2):
    # Gather all filenames from both directories
    all_filenames = []

    # Process first directory
    for filename in os.listdir(dir1):
        if filename.endswith('.csv'):
            # Ensuring only basename without extension (raw id) is left
            name_without_extension = os.path.splitext(os.path.basename(filename))[0]
            all_filenames.append(name_without_extension)

    # Process second directory
    for filename in os.listdir(dir2):
        if filename.endswith('.csv'):
            name_without_extension = os.path.splitext(os.path.basename(filename))[0]
            all_filenames.append(name_without_extension)

    # Remove duplicates
    unique_filenames = list(set(all_filenames))
    return unique_filenames


def split_data(data, train_size=0.7, val_size=0.15):
    # Split the data into train and temp (validation + test)
    train_data, temp_data = train_test_split(data, train_size=train_size, random_state=42)

    # Calculate the adjusted validation size from temp_data
    val_data, test_data = train_test_split(temp_data, test_size=(1 - val_size / (1 - train_size)), random_state=42)

    return train_data, val_data, test_data

def load_dataset(video_path, audio_path):
    # Load and union data
    combined_data = load_and_union_data(video_path, audio_path)

    # Split the data
    train_filenames, val_filenames, test_filenames = split_data(combined_data)

    # Create DataLoaders for each split (assuming the dataset class handles loading from the filenames)
    train_dataset = VideoAudioDataset(train_filenames, video_path, audio_path)
    val_dataset = VideoAudioDataset(val_filenames, video_path, audio_path)
    test_dataset = VideoAudioDataset(test_filenames, video_path, audio_path)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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
