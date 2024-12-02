import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from sklearn.model_selection import train_test_split
import argparse
from tqdm.auto import tqdm
import logging
from sklearn.metrics import accuracy_score


# Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--version", required=True, help="Version number")
args = parser.parse_args()

VERSION = args.version if args.version else "-1"

# Logging Configuration
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f"pretraining_model_{VERSION}.log")
# If the log file exists, delete it
if os.path.exists(log_filename):
    os.remove(log_filename)

# Configure logging to write to file and console
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for detailed logs
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
)

# Define global parameters
BLOCK_SIZE = 8
DRIFT_RANGE = 7
FRAME_RATE = 15
MFCC_FEATURES = 12  # mfcc_2 to mfcc_13
NUM_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
SUBSET_SIZE = 0.05  # Proportion of data to use

# param for number of random blocks to take per clip
BLOCKS_PER_CLIP = 10

# Model Definitions
DROPOUT = 0.5


class VisualNetwork(nn.Module):
    def __init__(self, block_size=BLOCK_SIZE):
        super(VisualNetwork, self).__init__()
        self.conv3d_1 = nn.Conv3d(1, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3d_1 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.dropout_1 = nn.Dropout3d(p=DROPOUT)

        self.conv3d_2 = nn.Conv3d(64, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.batch_norm_1 = nn.BatchNorm3d(32)
        self.pool3d_2 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.dropout_2 = nn.Dropout3d(p=DROPOUT)

        self.conv3d_3 = nn.Conv3d(32, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.batch_norm_2 = nn.BatchNorm3d(32)
        self.pool3d_3 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.dropout_3 = nn.Dropout3d(p=DROPOUT)

        # Calculate the flattened feature size after conv and pooling layers
        # Input: (1, BLOCK_SIZE, 32, 32)
        # After conv3d_1 and pool3d_1: (64, BLOCK_SIZE//2, 16, 16)
        # After conv3d_2 and pool3d_2: (32, BLOCK_SIZE//2, 8, 8)
        # After conv3d_3 and pool3d_3: (32, BLOCK_SIZE//2, 4, 4)
        self.fc = nn.Linear(32 * (BLOCK_SIZE // 2) * 4 * 4, 1024)

    def forward(self, x):
        # x shape: (batch_size, BLOCK_SIZE, 32, 32)
        x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 1, BLOCK_SIZE, 32, 32)
        x = F.relu(self.conv3d_1(x))
        x = self.pool3d_1(x)
        x = self.dropout_1(x)

        x = F.relu(self.conv3d_2(x))
        x = self.batch_norm_1(x)
        x = self.pool3d_2(x)
        x = self.dropout_2(x)

        x = F.relu(self.conv3d_3(x))
        x = self.batch_norm_2(x)
        x = self.pool3d_3(x)
        x = self.dropout_3(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc(x))
        return x


class AudioNetwork(nn.Module):
    def __init__(self, num_mfcc_rows, num_mfcc_features=MFCC_FEATURES):
        super(AudioNetwork, self).__init__()
        self.conv2d_1 = nn.Conv2d(1, 128, kernel_size=(2, 2), padding=(1, 1))
        self.pool2d_1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2d_2 = nn.Conv2d(128, 64, kernel_size=(2, 2), padding=(1, 1))
        self.batch_norm_1 = nn.BatchNorm2d(64)
        self.pool2d_2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2d_3 = nn.Conv2d(64, 32, kernel_size=(2, 2), padding=(1, 1))
        self.batch_norm_2 = nn.BatchNorm2d(32)
        self.pool2d_3 = nn.MaxPool2d(kernel_size=(2, 2))

        # Dynamically compute the flattened feature size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, num_mfcc_rows, num_mfcc_features)
            x = F.relu(self.conv2d_1(dummy_input))
            x = self.pool2d_1(x)
            x = F.relu(self.conv2d_2(x))
            x = self.batch_norm_1(x)
            x = self.pool2d_2(x)
            x = F.relu(self.conv2d_3(x))
            x = self.batch_norm_2(x)
            x = self.pool2d_3(x)
            flattened_size = x.numel()
            logging.info(f"AudioNetwork flattened size: {flattened_size}")

        self.fc = nn.Linear(flattened_size, 1024)

    def forward(self, x):
        # x shape: (batch_size, num_mfcc_rows, num_mfcc_features)
        logging.debug(f"Audio Input Shape: {x.shape}")
        x = x.unsqueeze(
            1
        )  # Add channel dimension: (batch_size, 1, num_mfcc_rows, num_mfcc_features)
        logging.debug(f"After unsqueeze: {x.shape}")
        x = F.relu(self.conv2d_1(x))
        logging.debug(f"After conv2d_1: {x.shape}")
        x = self.pool2d_1(x)
        logging.debug(f"After pool2d_1: {x.shape}")

        x = F.relu(self.conv2d_2(x))
        logging.debug(f"After conv2d_2: {x.shape}")
        x = self.batch_norm_1(x)
        x = self.pool2d_2(x)
        logging.debug(f"After pool2d_2: {x.shape}")

        x = F.relu(self.conv2d_3(x))
        logging.debug(f"After conv2d_2: {x.shape}")
        x = self.batch_norm_2(x)
        x = self.pool2d_3(x)
        logging.debug(f"After pool2d_2: {x.shape}")

        x = x.view(x.size(0), -1)  # Flatten
        logging.debug(f"After flatten: {x.shape}")
        x = F.relu(self.fc(x))
        logging.debug(f"After fc: {x.shape}")
        return x

# class PretrainingNetwork(nn.Module):
#     def __init__(self):
#         super(PretrainingNetwork, self).__init__()
#         self.fc1 = nn.Linear(2048, 1024)
#         self.fc2 = nn.Linear(1024, 1)
#
#     def forward(self, visual_feat, audio_feat):
#         x = torch.cat((visual_feat, audio_feat), dim=1)  # Concatenate features
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# Using below attention model
class PretrainingNetwork(nn.Module):
    def __init__(self, hidden_dim=1024):
        super(PretrainingNetwork, self).__init__()
        self.attention = CrossModalAttention(input_dim=1024, hidden_dim=1024, output_dim=1024)
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, visual_feat, audio_feat):
        attended_feat = self.attention(audio_feat, visual_feat)
        x = F.relu(self.fc1(attended_feat))
        x = self.fc2(x)
        return x


class CrossModalAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CrossModalAttention, self).__init__()
        self.query_fc = nn.Linear(input_dim, hidden_dim)
        self.key_fc = nn.Linear(input_dim, hidden_dim)
        self.value_fc = nn.Linear(input_dim, hidden_dim)
        self.output_fc = nn.Linear(hidden_dim, output_dim)  # Match visual feature dimension
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, audio_feat, visual_feat):
        query = self.query_fc(visual_feat)
        key = self.key_fc(audio_feat)
        value = self.value_fc(audio_feat)

        attention_weights = self.softmax(torch.matmul(query, key.transpose(-2, -1)))
        attended_audio = torch.matmul(attention_weights, value)
        attended_audio = self.output_fc(attended_audio)  # Project to 1024

        return attended_audio + visual_feat  # Residual connection


# class PretrainingAudioVisualModel(nn.Module):
#     def __init__(self, num_mfcc_rows, num_mfcc_features=MFCC_FEATURES):
#         super(PretrainingAudioVisualModel, self).__init__()
#         self.visual_network = VisualNetwork()
#         self.audio_network = AudioNetwork(num_mfcc_rows, num_mfcc_features)
#         self.fusion_network = PretrainingNetwork()
#
#     def forward(self, video_input, audio_input):
#         visual_feat = self.visual_network(video_input)
#         audio_feat = self.audio_network(audio_input)
#         output = self.fusion_network(visual_feat, audio_feat)
#         return output


class PretrainingAudioVisualModel(nn.Module):
    def __init__(self, num_mfcc_rows, num_mfcc_features=MFCC_FEATURES, hidden_dim=1024):
        super(PretrainingAudioVisualModel, self).__init__()
        self.visual_network = VisualNetwork()
        self.audio_network = AudioNetwork(num_mfcc_rows, num_mfcc_features)

        # Attention module for cross-modal fusion
        self.attention = CrossModalAttention(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim)

        # Output layer
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, video_input, audio_input):
        # Extract visual and audio features
        visual_feat = self.visual_network(video_input)
        audio_feat = self.audio_network(audio_input)

        # Apply attention between audio and visual features
        attended_feat = self.attention(audio_feat, visual_feat)

        # Forward the attended features through the fully connected layers
        x = F.relu(self.fc1(attended_feat))
        x = self.fc2(x)

        return x



# Dataset Definition


class AudioVisualSyncDataset(Dataset):
    def __init__(
        self,
        video_ids,
        audio_dir,
        visual_dir,
        block_size=BLOCK_SIZE,
        drift_range=DRIFT_RANGE,
        frame_rate=FRAME_RATE,
        mfcc_increment=None,
        step_size=1,
        preload_data=False,
    ):
        self.video_ids = video_ids
        self.audio_dir = audio_dir
        self.visual_dir = visual_dir
        self.block_size = block_size
        self.drift_range = drift_range
        self.shift_range = np.concatenate((np.arange(-12, -4), np.arange(5, 13)))    # from -12 to -5 (exclusive of -4)

        self.frame_rate = frame_rate
        self.step_size = step_size
        self.preload_data = preload_data

        self.frame_duration_ms = 1000 / self.frame_rate

        if mfcc_increment is None:
            self.mfcc_increment_ms = (1 / (self.frame_rate * self.drift_range)) * 1000
        else:
            self.mfcc_increment_ms = mfcc_increment

        self.samples = []
        self.preloaded_data = {}

        self.prepare_samples()

    def prepare_samples(self):
        for video_id in tqdm(self.video_ids, desc="Videos", leave=False):
            audio_file = os.path.join(self.audio_dir, f"{video_id}.csv")
            visual_file = os.path.join(self.visual_dir, f"{video_id}.parquet")

            if not os.path.exists(audio_file) or not os.path.exists(visual_file):
                logging.warning(f"Missing files for video_id {video_id}. Skipping.")
                continue  # Skip if either file does not exist

            try:
                audio_df = pd.read_csv(audio_file)
                audio_df = audio_df.rename(columns={"video_number": "clip_num"})
                visual_df = pd.read_parquet(visual_file)
                # Ensure both clip_num columns are integers
                audio_df["clip_num"] = audio_df["clip_num"].astype(int)
                visual_df["clip_num"] = visual_df["clip_num"].astype(int)
            except Exception as e:
                logging.error(f"Error loading files for video_id {video_id}: {e}")
                continue

            clip_keys = visual_df[["video_id", "clip_num", "desync"]].drop_duplicates()

            for _, clip in clip_keys.iterrows():
                clip_video_id = clip["video_id"]
                clip_num = clip["clip_num"]
                desync = int(clip['desync'])

                video_clip_df = visual_df[
                    (visual_df["video_id"] == clip_video_id)
                    & (visual_df["clip_num"] == clip_num)
                ].reset_index(drop=True)
                audio_clip_df = audio_df[
                    (audio_df["video_id"] == clip_video_id)
                    & (audio_df["clip_num"] == clip_num)
                ].reset_index(drop=True)

                if video_clip_df.empty and audio_clip_df.empty:
                    logging.warning(
                        f"Empty visual and audio clips for video_id {video_id}, clip_num {clip_num}. Skipping."
                    )
                    continue

                if video_clip_df.empty:
                    logging.warning(
                        f"Empty visual clip for video_id {video_id}, clip_num {clip_num}. Skipping."
                    )
                    continue

                if audio_clip_df.empty:
                    logging.warning(
                        f"Empty audio clip for video_id {video_id}, clip_num {clip_num}. Skipping."
                    )
                    continue

                # logging.info(
                #     f"Processing video_id {video_id}, clip_num {clip_num} with {len(video_clip_df)} frames and {len(audio_clip_df)} MFCC rows."
                # )

                total_frames = video_clip_df["frame_number"].max() + 1
                total_mfcc_rows = len(audio_clip_df)

                total_duration_ms = total_frames * self.frame_duration_ms

                fixed_num_mfcc_rows = int(
                    np.round(
                        (self.block_size * self.frame_duration_ms)
                        / self.mfcc_increment_ms
                    )
                )

                # Randomly select certain number of clips
                for i in range(BLOCKS_PER_CLIP):
                    random_start_frame = random.randint(0, total_frames - self.block_size)

                    # Randomly determine if synced or not
                    is_desynchronized = random.choice([0, 1])  # 0 = synchronized, 1 = desynchronized

                    # Create the label
                    if is_desynchronized:
                        label = np.zeros(1, dtype=np.float32)
                        desync_amount = np.random.choice(self.shift_range)
                    else:
                        label = np.ones(1, dtype=np.float32)
                        desync_amount = 0

                    video_block_df = video_clip_df.iloc[random_start_frame : random_start_frame + self.block_size]
                    start_time_ms = (-desync + random_start_frame + desync_amount) * self.frame_duration_ms
                    end_time_ms = start_time_ms + self.block_size * self.frame_duration_ms

                    if start_time_ms < 0 or end_time_ms > total_duration_ms:
                        continue

                    mfcc_start_idx = int(
                        np.round(start_time_ms / self.mfcc_increment_ms)
                    )
                    mfcc_end_idx = mfcc_start_idx + fixed_num_mfcc_rows

                    if mfcc_start_idx < 0 or mfcc_end_idx > total_mfcc_rows:
                        continue

                    audio_block_df = audio_clip_df.iloc[mfcc_start_idx:mfcc_end_idx]

                    if len(audio_block_df) != fixed_num_mfcc_rows:
                        continue

                    video_features = video_block_df.filter(
                        regex="^feature_"
                    ).values.astype(np.float32)
                    audio_features = audio_block_df.filter(
                        regex="^mfcc_"
                    ).values.astype(np.float32)

                    sample = {
                        "video_features": video_features,
                        "audio_features": audio_features,
                        "label": label,
                    }

                    if self.preload_data:
                        key = f"{video_id}_{clip_num}_{i}_{label}"
                        self.preloaded_data[key] = sample
                        self.samples.append(key)
                    else:
                        self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.preload_data:
            key = self.samples[idx]
            sample = self.preloaded_data[key]
        else:
            sample = self.samples[idx]

        video_features = sample["video_features"]
        audio_features = sample["audio_features"]
        label = sample["label"]

        try:
            # Reshape video features from (BLOCK_SIZE, 1024) to (BLOCK_SIZE, 32, 32)
            video_features = video_features.reshape(BLOCK_SIZE, 32, 32)
        except Exception as e:
            logging.error(f"Error reshaping video features: {e}")
            raise e

        video_tensor = torch.from_numpy(video_features)  # Shape: (BLOCK_SIZE, 32, 32)
        audio_tensor = torch.from_numpy(
            audio_features
        )  # Shape: (fixed_num_mfcc_rows, 12)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return video_tensor, audio_tensor, label_tensor


# Utility Functions


def get_common_filenames(audio_dir, visual_dir, prop=None):
    audio_files = set(f[:-4] for f in os.listdir(audio_dir) if f.endswith(".csv"))
    visual_files = set(f[:-8] for f in os.listdir(visual_dir) if f.endswith(".parquet"))
    common_files = list(audio_files.intersection(visual_files))
    if prop:
        if prop > 1.0 or prop < 0.0:
            logging.error("Subset size prop must be between 0 and 1.")
            return []
        subset_size = int(prop * len(common_files))
        common_files = random.sample(common_files, subset_size)
    return common_files


def split_filenames(common_files, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    train_files, temp_files = train_test_split(
        common_files, train_size=train_ratio, random_state=42
    )
    val_files, test_files = train_test_split(
        temp_files, test_size=test_ratio / (test_ratio + val_ratio), random_state=42
    )
    return train_files, val_files, test_files


def write_filenames(filename, file_list):
    with open(filename, "w") as f:
        for item in file_list:
            f.write(f"{item}\n")


def read_filenames(filename):
    with open(filename, "r") as f:
        return [line.strip() for line in f]


# Main Training Script


def main():
    audio_dir = "/gpfs2/classes/cs6540/AVSpeech/5-2_audio/train/"
    visual_dir = "/gpfs2/classes/cs6540/AVSpeech/6-2_visual_features/train_optical/"

    # Check if split files exist
    if (
        not os.path.exists(f"all_filenames_{VERSION}.txt")
        or not os.path.exists(f"train_filenames_{VERSION}.txt")
        or not os.path.exists(f"val_filenames_{VERSION}.txt")
        or not os.path.exists(f"test_filenames_{VERSION}.txt")
    ):

        # Get common filenames with SUBSET_SIZE proportion
        common_files = get_common_filenames(audio_dir, visual_dir, SUBSET_SIZE)
        if not common_files:
            logging.error("No common files found. Exiting.")
            return

        write_filenames(f"all_filenames_{VERSION}.txt", common_files)

        # Split filenames
        train_files, val_files, test_files = split_filenames(common_files)
        write_filenames(f"train_filenames_{VERSION}.txt", train_files)
        write_filenames(f"val_filenames_{VERSION}.txt", val_files)
        write_filenames(f"test_filenames_{VERSION}.txt", test_files)

        logging.info(
            f"Generated train, val, test splits with SUBSET_SIZE={SUBSET_SIZE}"
        )
    else:
        common_files = read_filenames(f"all_filenames_{VERSION}.txt")
        train_files = read_filenames(f"train_filenames_{VERSION}.txt")
        val_files = read_filenames(f"val_filenames_{VERSION}.txt")
        test_files = read_filenames(f"test_filenames_{VERSION}.txt")

        logging.info("Loaded existing train, val, test splits.")

    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Initialize model parameters
    fixed_num_mfcc_rows = int(
        np.round(
            (BLOCK_SIZE * (1000 / FRAME_RATE))
            / ((1 / (FRAME_RATE * DRIFT_RANGE)) * 1000)
        )
    )
    logging.info(f"Fixed number of MFCC rows per block: {fixed_num_mfcc_rows}")

    # Instantiate the model
    model = PretrainingAudioVisualModel(num_mfcc_rows=fixed_num_mfcc_rows).to(device)
    logging.info("Model instantiated.")

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    losses = []

    for epoch in tqdm(range(NUM_EPOCHS), desc="Epochs"):
        logging.info(f"Starting Epoch {epoch+1}/{NUM_EPOCHS}")
        all_predictions = []
        all_labels = []

        # Randomly sample 25% of the train files
        sample_size = max(1, int(len(train_files) * 0.25))
        train_sample_files = random.sample(train_files, sample_size)
        logging.info(f"Epoch {epoch+1}: Training on {sample_size} files.")

        # Create training dataset and dataloader
        train_dataset = AudioVisualSyncDataset(
            video_ids=train_sample_files,
            audio_dir=audio_dir,
            visual_dir=visual_dir,
            preload_data=True,
            step_size=8
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
        )

        # Training loop with tqdm
        model.train()
        total_train_loss = 0
        with tqdm(
            total=len(train_dataloader), desc="Training Batches", leave=False
        ) as pbar:
            for video_batch, audio_batch, labels in train_dataloader:
                video_batch = video_batch.to(device)
                audio_batch = audio_batch.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(video_batch, audio_batch)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item() * video_batch.size(0)
                pbar.update(1)

        avg_train_loss = total_train_loss / len(train_dataset)
        logging.info(f"Epoch {epoch+1}: Training Loss: {avg_train_loss:.4f}")

        # Randomly sample 25% of the test files
        test_sample_size = max(1, int(len(test_files) * 0.25))
        test_sample_files = random.sample(test_files, test_sample_size)
        logging.info(f"Epoch {epoch+1}: Evaluating on {test_sample_size} test files.")

        # Create testing dataset and dataloader
        test_dataset = AudioVisualSyncDataset(
            video_ids=test_sample_files,
            audio_dir=audio_dir,
            visual_dir=visual_dir,
            preload_data=False,
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
        )

        # Evaluation loop with tqdm
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            with tqdm(
                total=len(test_dataloader), desc="Validation Batches", leave=False
            ) as pbar:
                for video_batch, audio_batch, labels in test_dataloader:
                    video_batch = video_batch.to(device)
                    audio_batch = audio_batch.to(device)
                    labels = labels.to(device)

                    outputs = model(video_batch, audio_batch)
                    loss = criterion(outputs, labels)

                    total_test_loss += loss.item() * video_batch.size(0)
                    losses.append(torch.sqrt(loss).item())

                    # predictions = torch.sigmoid(outputs)  # Apply sigmoid - doesn't really make sense since non-negative values
                    # predicted_labels = (predictions >= 0.5).long()
                    predicted_labels = (outputs >= 0.5).long()

                    # Store the predicted and true labels for accuracy calculation
                    all_predictions.extend(predicted_labels.cpu().numpy())  # Convert to numpy for accuracy_score
                    all_labels.extend(labels.cpu().numpy())

                pbar.update(1)

        avg_test_loss = total_test_loss / len(test_dataset)
        logging.info(f"Epoch {epoch+1}: Validation Loss: {avg_test_loss:.4f}")
        accuracy = accuracy_score(all_labels, all_predictions)
        logging.info(f"Epoch {epoch+1}: Accuracy: {accuracy:.4f}")


        # Save the best model
        if epoch == 0:
            best_loss = total_test_loss
            torch.save(
                model.state_dict(), f"audio_visual_sync_model_{VERSION}_best.pth"
            )
            logging.info(f"Epoch {epoch+1}: Best model saved.")
        elif total_test_loss < best_loss:
            best_loss = total_test_loss
            torch.save(
                model.state_dict(), f"audio_visual_sync_model_{VERSION}_best.pth"
            )
            logging.info(
                f"Epoch {epoch+1}: Improved validation loss. Best model updated."
            )

    # Save the final model
    torch.save(model.state_dict(), f"audio_visual_sync_model_{VERSION}.pth")
    logging.info(
        f"Training complete. Final model saved as 'audio_visual_sync_model_{VERSION}.pth'."
    )

    # Optionally, save RMSEs if needed
    rmses = np.array(losses)
    np.save(f"rmse_history_{VERSION}.npy", rmses)
    logging.info(f"RMSE history saved as 'rmse_history_{VERSION}.npy'.")


if __name__ == "__main__":
    main()
