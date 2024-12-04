from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import VisualNetwork, AudioNetwork, AudioVisualModel, AudioVisualSyncDataset, FusionNetwork, \
    write_filenames, get_common_filenames, split_filenames
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import argparse
import random
from tqdm.auto import tqdm

# Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--version", required=True, help="Version number")
args = parser.parse_args()

VERSION = args.version if args.version else "-1"

# Logging Configuration
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f"evaluate_model_{VERSION}.log")
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
LEARNING_RATE = 1e-3
SUBSET_SIZE = 0.05  # Proportion of data to use


def plot_rmse_distribution(rmse_distribution, filename='rme_dist.png'):
    # Sort the RMSE distribution by input values for better visualization
    input_values = list(rmse_distribution.keys())
    rmse_values = list(rmse_distribution.values())

    # Create a bar plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x=input_values, y=rmse_values, palette="viridis")

    plt.xlabel('Input Values', fontsize=14)
    plt.ylabel('RMSE', fontsize=14)
    plt.title('RMSE Distribution Across Input Values', fontsize=16)
    plt.xticks(rotation=45)  # Rotate x labels for better readability
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def evaluate_model(model, test_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    all_preds = []
    all_targets = []

    grouped_predictions = defaultdict(list)
    grouped_targets = defaultdict(list)

    with torch.no_grad():  # Disable gradient calculations
        for features, attention_mask, target in test_loader:
            features.to(device)
            attention_mask.to(device)
            target.to(device)

            outputs = model(features, src_mask=attention_mask)
            loss = criterion(outputs.view(-1), target.view(-1))
            total_loss += loss.item()

            outputs_np = outputs.cpu().numpy()
            target_np = target.cpu().numpy()

            all_preds.extend(outputs_np)  # Store predictions
            all_targets.extend(target_np)  # Store targets

            # Group predictions and targets by input value
            for pred, true in zip(outputs_np, target_np):
                grouped_predictions[true].append(pred)
                grouped_targets[true].append(true)

    avg_test_loss = total_loss / len(test_loader)
    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_preds)

    # Calculate RMSE for each unique input value
    rmse_distribution = {}
    for inp in grouped_predictions:
        if len(grouped_predictions[inp]) > 0:  # Avoid division by zero
            rmse_ = np.sqrt(mean_squared_error(grouped_targets[inp], grouped_predictions[inp]))
            rmse_distribution[inp] = rmse_

    return avg_test_loss, mse, rmse, mae, rmse_distribution


def read_filenames(filename):
    with open(filename, 'r') as file:
        filenames = [line.strip() for line in file]
    return filenames


def main():
    # Load model - Should mirror the training
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
    model = AudioVisualModel(num_mfcc_rows=fixed_num_mfcc_rows).to(device)
    model_name = "audio_visual_sync_model_5_best.pth"
    model.load_state_dict(torch.load(model_name))
    logging.info("Model instantiated.")

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    rmses = []

    # Randomly sample 25% of the test files
    test_sample_size = max(1, int(len(val_files) * 0.25))
    test_sample_files = random.sample(val_files, test_sample_size)
    logging.info(f"Evaluating on {test_sample_size} test files.")

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
    all_preds = []
    all_targets = []

    grouped_predictions = defaultdict(list)
    grouped_targets = defaultdict(list)

    with torch.no_grad():
        with tqdm(
                total=len(test_dataloader), desc="Validation Batches", leave=False
        ) as pbar:
            for video_batch, audio_batch, labels in test_dataloader:
                video_batch = video_batch.to(device)
                audio_batch = audio_batch.to(device)
                labels = labels.to(device)

                outputs = model(video_batch, audio_batch)
                # loss = criterion(outputs.squeeze(), labels)
                if outputs.shape != labels.shape:
                    if outputs.shape[-1] == 1 and len(
                            outputs.shape) > 1:  # Handle case where outputs have an extra dimension
                        outputs = outputs.squeeze(-1)
                    elif len(labels.shape) == 1:  # Handle case where labels need an extra dimension
                        labels = labels.unsqueeze(-1)
                loss = criterion(outputs, labels)

                total_test_loss += loss.item() * video_batch.size(0)
                rmses.append(torch.sqrt(loss).item())

                outputs_np = outputs.cpu().numpy()
                target_np = labels.cpu().numpy()

                all_preds.extend(outputs_np)  # Store predictions
                all_targets.extend(target_np)  # Store targets

                # Group predictions and targets by input value
                for pred, true in zip(outputs_np, target_np):
                    grouped_predictions[true].append(pred)
                    grouped_targets[true].append(true)

    avg_test_loss = total_test_loss / len(test_dataloader)
    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_preds)
    print(f"Average test loss: {avg_test_loss}\n"
          f"MSE: {mse}\n"
          f"RMSE: {rmse}\n"
          f"MAE: {mae}")

    # Calculate RMSE for each unique input value
    rmse_distribution = {}
    for inp in grouped_predictions:
        if len(grouped_predictions[inp]) > 0:  # Avoid division by zero
            rmse_ = np.sqrt(mean_squared_error(grouped_targets[inp], grouped_predictions[inp]))
            rmse_distribution[inp] = rmse_

    plt.figure(figsize=(10, 6))
    sns.histplot(all_preds, kde=True, color='blue', label='Predictions', bins=30)
    plt.xlabel('Predicted Values')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predictions')
    plt.legend()
    plt.savefig(f"all_pred_hist_{VERSION}.png")

    plot_rmse_distribution(rmse_distribution)


if __name__=="__main__":
    main()
