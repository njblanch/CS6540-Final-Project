from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from custom_transformer import MultimodalTransformer
from data_loading import VideoAudioDataset
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


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


if __name__=="__main__":
    # Load model - Should mirror the training
    audio_dim = 118
    video_dim = 128 #1280
    n_heads = 6  # Needs to be d_model/n_heads is an int
    num_layers = 6  # Change as needed
    d_model = audio_dim + video_dim  # audio_dim + video_dim
    dim_feedforward = 256  # Change as needed

    MODEL_NAME = 'transformer_desync_model_v128_100_256_best.pth' #'transformer_desync_model_v128_2_256_best.pth'

    # Load data
    audio_path = "/gpfs2/classes/cs6540/AVSpeech/5_audio/train"
    video_path = "/gpfs2/classes/cs6540/AVSpeech/6_visual_features/train_dist_128"
    test_filenames = read_filenames("test_filenames.txt")
    test_dataset = VideoAudioDataset(test_filenames, video_path, audio_path)

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # Initialize model, optimizer, and loss function
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model = MultimodalTransformer(features_dim=d_model, nhead=n_heads, num_layers=num_layers, d_model=d_model,
                                  output_dim=1, device=device)
    model.load_state_dict(torch.load(MODEL_NAME))

    criterion = nn.MSELoss()

    avg_test_loss, mse, rmse, mae, rmse_dist = evaluate_model(model, test_loader, criterion, device)
    print(f"Validation Loss: {avg_test_loss:.4f}, MSE: {mse:.4f}, rMSE: {rmse:.4f}, MAE: {mae:.4f}")

    plot_rmse_distribution(rmse_dist)

