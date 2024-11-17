# transformer_training.py

import argparse
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, r2_score

from custom_transformer import MultiModalTransformer  # Ensure this is correct
from data_loading import load_dataset

# Parse command-line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--version", required=True, help="Version number")
args = ap.parse_args()
VERSION = args.version


# Function for variance regularization
def variance_regularization(predictions, alpha):
    variance = torch.var(predictions)
    return alpha * (1.0 / (variance + 1e-6))


def verify_unique_video_ids(train_dataset, val_dataset, test_dataset):
    train_ids = set([clip[0] for clip in train_dataset.clip_list])
    val_ids = set([clip[0] for clip in val_dataset.clip_list])
    test_ids = set([clip[0] for clip in test_dataset.clip_list])

    overlap_train_val = train_ids.intersection(val_ids)
    overlap_train_test = train_ids.intersection(test_ids)
    overlap_val_test = val_ids.intersection(test_ids)

    assert not overlap_train_val, f"Overlap between train and val: {overlap_train_val}"
    assert (
        not overlap_train_test
    ), f"Overlap between train and test: {overlap_train_test}"
    assert not overlap_val_test, f"Overlap between val and test: {overlap_val_test}"
    print("Verification passed: No overlapping video IDs across splits.")


# Hyperparameters
EPOCHS = 10
ALPHA_START = 1
ALPHA_END = 0.25
alphas = torch.linspace(ALPHA_START, ALPHA_END, EPOCHS)
batch_size = 16
learning_rate = 1e-4
weight_decay = 1e-5

audio_dim = 120
video_dim = 1024
n_heads_transformer = 8
num_layers = 2
d_model = 384  # Ensure d_model is divisible by n_heads_transformer (384 / 8 = 48)
dim_feedforward = 256
output_dim = 1
max_seq_length = 225  # 15 fps * 15 seconds
dropout = 0.1

# Paths and dataset parameters
final_model_path = f"transformer_desync_model_final_v{VERSION}.pth"
best_model_path = f"transformer_desync_model_best_v{VERSION}.pth"

# Update paths as per your environment
audio_path = "../5_audio/train"
video_path = "../6-1_visual_features/train_1024_cae"
normalization_params_path = "normalization_params_cae.pth"

max_data = {"train": 1000, "val": 200, "test": 200}

# Load datasets
train_dataset, val_dataset, test_dataset = load_dataset(
    video_path=video_path,
    audio_path=audio_path,
    normalization_params_path=normalization_params_path,
    data_sizes=max_data,
    save=True,
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

verify_unique_video_ids(train_dataset, val_dataset, test_dataset)

print("Dataset Sizes:")
print(f"Train: {len(train_loader.dataset)} samples")
print(f"Validation: {len(val_loader.dataset)} samples")
print(f"Test: {len(test_loader.dataset)} samples")

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model
model = MultiModalTransformer(
    video_dim=video_dim,
    audio_dim=audio_dim,
    n_heads_transformer=n_heads_transformer,
    num_layers=num_layers,
    d_model=d_model,
    dim_feedforward=dim_feedforward,
    output_dim=output_dim,
    max_seq_length=max_seq_length,
    dropout=dropout,
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

# Move alphas to device
alphas = alphas.to(device)

# Training Loop
for epoch in tqdm(range(EPOCHS), desc="Epochs"):
    model.train()
    total_loss = 0.0
    total_primary_loss = 0.0
    total_reg_loss = 0.0

    for batch in train_loader:
        try:
            video_features, audio_features, actual_length, y_offset = batch

            video_features = video_features.to(device)
            audio_features = audio_features.to(device)
            y_offset = y_offset.to(device).float()

            batch_size_current = video_features.size(0)
            seq_len = video_features.size(1)

            # Create key_padding_mask based on actual sequence lengths
            key_padding_mask = torch.zeros(
                batch_size_current, seq_len, dtype=torch.bool, device=device
            )
            for i, length in enumerate(actual_length):
                if length < seq_len:
                    key_padding_mask[i, length:] = True  # True indicates padding

            # Generate positions for positional encoding
            positions = torch.zeros(
                batch_size_current, seq_len, dtype=torch.long, device=device
            )
            for i, length in enumerate(actual_length):
                positions[i, :length] = torch.arange(length, device=device)

            optimizer.zero_grad()

            # Forward pass
            output = model(
                video_features,
                audio_features,
                video_positions=positions,
                audio_positions=positions,
                key_padding_mask=key_padding_mask,
            ).view(-1)

            # Compute losses
            primary_loss = criterion(output, y_offset)
            reg_loss = variance_regularization(output, alphas[epoch])
            loss = primary_loss + reg_loss

            # Accumulate losses
            total_loss += loss.item()
            total_primary_loss += primary_loss.item()
            total_reg_loss += reg_loss.item()

            # Backpropagation
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
        except IndexError:
            # Skip problematic samples
            continue
        except Exception as e:
            print(f"Error during training at epoch {epoch + 1}: {e}", flush=True)
            continue

    # Average losses
    avg_loss = total_loss / len(train_loader)
    avg_primary_loss = total_primary_loss / len(train_loader)
    avg_reg_loss = total_reg_loss / len(train_loader)

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}, Total Loss: {avg_loss:.4f}, "
        f"Primary Loss: {avg_primary_loss:.4f}, Reg Loss: {avg_reg_loss:.4f}, "
        f"RMSE: {avg_primary_loss ** 0.5:.4f}"
    )

    # Validation Loop
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            try:
                video_features, audio_features, actual_length, y_offset = batch

                video_features = video_features.to(device)
                audio_features = audio_features.to(device)
                y_offset = y_offset.to(device).float()

                batch_size_current = video_features.size(0)
                seq_len = video_features.size(1)

                # Create key_padding_mask for validation
                key_padding_mask = torch.zeros(
                    batch_size_current, seq_len, dtype=torch.bool, device=device
                )
                for i, length in enumerate(actual_length):
                    if length < seq_len:
                        key_padding_mask[i, length:] = True  # True indicates padding

                # Generate positions for positional encoding
                positions = torch.zeros(
                    batch_size_current, seq_len, dtype=torch.long, device=device
                )
                for i, length in enumerate(actual_length):
                    positions[i, :length] = torch.arange(length, device=device)

                # Forward pass
                outputs = model(
                    video_features,
                    audio_features,
                    video_positions=positions,
                    audio_positions=positions,
                    key_padding_mask=key_padding_mask,
                ).view(-1)

                # Compute loss
                loss = criterion(outputs, y_offset)
                total_val_loss += loss.item()
            except IndexError:
                # Skip problematic samples
                continue
            except Exception as e:
                print(f"Error during validation at epoch {epoch + 1}: {e}", flush=True)
                continue

    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}, RMSE: {avg_val_loss ** 0.5:.4f}")
    scheduler.step(avg_val_loss)

    # Save the best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)
        print(
            f"Saved best model with validation loss: {best_val_loss:.4f}, RMSE: {best_val_loss ** 0.5:.4f}"
        )

# Save the final model
torch.save(model.state_dict(), final_model_path)
print("Successfully finished training!")
