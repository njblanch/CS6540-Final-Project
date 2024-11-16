# transformer_training.py

import argparse
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, r2_score

from custom_3DCNN import AVsync_CNN
from data_loading import load_dataset

# Parse command-line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--version", required=True, help="Version number")
args = ap.parse_args()
VERSION = args.version

# Hyperparameters
EPOCHS = 250
ALPHA_START = 1
ALPHA_END = 0
alphas = torch.linspace(ALPHA_START, ALPHA_END, EPOCHS)
batch_size = 64
learning_rate = 1e-4
weight_decay = 1e-5

audio_dim = (118)
video_dim = (8, 8, 16)
output_dim = 1
max_seq_length = 225  # 15 fps * 15 seconds

# Paths and dataset parameters
final_model_path = f"transformer_desync_model_final_v{VERSION}.pth"
best_model_path = f"transformer_desync_model_best_v{VERSION}.pth"

audio_path = "/gpfs2/classes/cs6540/AVSpeech/5_audio/train"
video_path = "/gpfs2/classes/cs6540/AVSpeech/6-1_visual_features/train_1024_cae"
normalization_params_path = "normalization_params_cae.pth"

max_data = {"train": 10000, "val": 2000, "test": 2000}
# max_data = {"train": 100, "val": 20, "test": 20}

# Load datasets
train_loader, val_loader, test_loader = load_dataset(
    video_path=video_path,
    audio_path=audio_path,
    normalization_params_path=normalization_params_path,
    max_data=max_data,
    save=True,
)
print("Dataset Sizes:")
print(f"Train: {len(train_loader.dataset)} samples")
print(f"Validation: {len(val_loader.dataset)} samples")
print(f"Test: {len(test_loader.dataset)} samples")

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

# Initialize model
model = AVsync_CNN(
    video_dim=video_dim,
    audio_dim=audio_dim,
).to(device)

# Initialize optimizer and loss function
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=weight_decay
)
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.1, patience=3
)
best_val_loss = float("inf")


# Function for variance regularization
def variance_regularization(predictions, alpha):
    variance = torch.var(predictions)
    return alpha * (1.0 / (variance + 1e-6))


# Training Loop
for epoch in tqdm(range(EPOCHS), desc="Epochs"):
    model.train()
    total_loss = 0.0
    total_primary_loss = 0.0
    total_reg_loss = 0.0

    for batch in train_loader:
        video_features, audio_features, actual_length, y_offset = batch

        video_features = video_features.to(device)
        audio_features = audio_features.to(device)
        y_offset = y_offset.to(device).float()

        batch_size = video_features.size(0)
        seq_len = video_features.size(1)

        # Create key_padding_mask based on actual sequence lengths
        key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        for i, length in enumerate(actual_length):
            if length < seq_len:
                key_padding_mask[i, length:] = True  # True indicates padding

        # Forward pass
        output = model(
            video_features,
            audio_features,
            visual_mask=key_padding_mask,
            audio_mask=key_padding_mask,
        ).view(-1)

        # Compute losses
        primary_loss = criterion(output, y_offset)
        reg_loss = variance_regularization(output, alpha=alphas[epoch])
        loss = primary_loss + reg_loss

        # Accumulate losses
        total_loss += loss.item()
        total_primary_loss += primary_loss.item()
        total_reg_loss += reg_loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

    # Average losses
    avg_loss = total_loss / len(train_loader)
    avg_primary_loss = total_primary_loss / len(train_loader)
    avg_reg_loss = total_reg_loss / len(train_loader)

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}, Total Loss: {avg_loss:.4f}, "
        f"Primary Loss: {avg_primary_loss:.4f}, Reg Loss: {avg_reg_loss:.4f}, "
        f"RMSE: {avg_primary_loss ** 0.5:.4f}", flush=True
    )

    # Validation Loop
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            video_features, audio_features, actual_length, y_offset = batch

            video_features = video_features.to(device)
            audio_features = audio_features.to(device)
            y_offset = y_offset.to(device).float()

            batch_size = video_features.size(0)
            seq_len = video_features.size(1)
            print(batch_size)

            # Create key_padding_mask for validation
            key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
            for i, length in enumerate(actual_length):
                if length < seq_len:
                    key_padding_mask[i, length:] = True  # True indicates padding

            # Forward pass
            outputs = model(
                video_features,
                audio_features,
                visual_mask=key_padding_mask,
                audio_mask=key_padding_mask,
            ).view(-1)

            # Compute loss
            loss = criterion(outputs, y_offset)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}, RMSE: {avg_val_loss ** 0.5:.4f}")
    scheduler.step(avg_val_loss)

    # Save the best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)
        print(
            f"Saved best model with validation loss: {best_val_loss:.4f}, RMSE: {best_val_loss ** 0.5:.4f}", flush=True
        )

# Save the final model
torch.save(model.state_dict(), final_model_path)
print("Successfully finished training!")
