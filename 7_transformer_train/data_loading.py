import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader


class VideoAudioDataset(Dataset):
    # This provides all features per video_id and clip number - will have to change data loading things
    def __init__(self, filenames, video_path, audio_path):
        self.audio_features_list = []
        self.video_features_list = []
        self.y_offset_list = []
        self.video_id_list = []
        self.clip_number_list = []

        for filename in filenames:
            audio_csv = os.path.join(audio_path, filename + '.csv')  # Assuming CSV for audio
            video_csv = os.path.join(video_path, filename + '.csv')  # Assuming CSV for video
            self.process_data(audio_csv, video_csv)

    def process_data(self, audio_csv, video_csv):
        if os.path.exists(audio_csv) and os.path.exists(video_csv):
            audio_data = pd.read_csv(audio_csv)
            video_data = pd.read_csv(video_csv)

            # Group using video_id and clip_number
            audio_groups = audio_data.groupby(['video_id', 'clip_number'])
            video_groups = video_data.groupby(['video_id', 'clip_number'])

            # Process audio features
            for (video_id, clip_number), audio_group in audio_groups:
                if (video_id, clip_number) in video_groups.groups:
                    video_group = video_groups.get_group((video_id, clip_number))

                    # Ensure the number of rows match
                    if len(audio_group) == len(video_group):
                        audio_features = torch.tensor(
                            audio_group.drop(columns=['video_id', 'video_number', 'frame_number']).values)  # All audio features
                        video_features = torch.tensor(video_group.drop(
                            columns=['video_id', 'video_number', 'frame_number', 'desync']).values)  # All video features
                        y_offset = torch.tensor(video_group['desync'].values)  # Target variable

                        # Extend lists
                        self.audio_features_list.extend(audio_features)
                        self.video_features_list.extend(video_features)
                        self.y_offset_list.extend(y_offset)
                        self.video_id_list.extend(video_id)
                        self.clip_number_list.extend(clip_number)

    def __len__(self):
        return len(self.audio_features_list)

    def __getitem__(self, idx):
        video_id = self.video_id_list[idx]
        clip_number = self.clip_number_list[idx]

        audio_features = self.audio_features_list[idx] if idx < len(self.audio_features_list) else None
        video_features = self.video_features_list[idx] if idx < len(self.video_features_list) else None
        y_offset = self.y_offset_list[idx] if idx < len(self.y_offset_list) else None

        return video_id, clip_number, audio_features, video_features, y_offset


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
            file_path = os.path.join(dir2, filename)  # Full path for size check
            if os.path.getsize(file_path) > 1922: # Ensure file has things in it
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
    pass