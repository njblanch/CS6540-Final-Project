# transformer_training.py

from custom_transformer import MultiModalTransformer

# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from data_loading import load_dataset  # Ensure this is correctly imported
from tqdm.auto import tqdm
from sklearn.metrics import mean_absolute_error, r2_score
import argparse

# one input is VERSION
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--version", required=True, help="Version number")
args = ap.parse_args()
VERSION = args.version


ALPHA_START = 1
ALPHA_END = 0
EPOCHS = 250
alphas = torch.linspace(ALPHA_START, ALPHA_END, EPOCHS)


def variance_regularization(predictions, alpha):
    variance = torch.var(predictions)
    # Regularization term to encourage higher variance
    return alpha * (1.0 / (variance + 1e-6))


if __name__ == "__main__":
    # writer = SummaryWriter()
    # Paths and loading data
    final_model_path = f"transformer_desync_model100_final_small_v{VERSION}.pth"
    best_model_path = f"transformer_desync_model100_best_small_v{VERSION}.pth"

    audio_path = "/gpfs2/classes/cs6540/AVSpeech/5_audio/train"
    video_path = "/gpfs2/classes/cs6540/AVSpeech/6_visual_features/train_dist_l_256_256"
    normalization_params_path = "normalization_params.pth"

    max_data = {"train": 10000, "val": 2000, "test": 2000}

    train_loader, val_loader, test_loader = load_dataset(
        video_path=video_path,
        audio_path=audio_path,
        normalization_params_path=normalization_params_path,
        max_data=max_data,
        save=True,
    )
    print("Dataset Sizes:", flush=True)
    print(f"Train: {len(train_loader.dataset)} samples")
    print(f"Validation: {len(val_loader.dataset)} samples")
    print(f"Test: {len(test_loader.dataset)} samples")

    # Hyperparameters
    num_epochs = EPOCHS
    batch_size = 64
    learning_rate = 1e-4

    audio_dim = 120
    video_dim = 256
    n_heads_video = 4
    n_heads_audio = 2
    n_heads_transformer = 8
    num_layers = 4
    d_model = audio_dim + video_dim  # 120 + 256 = 376
    dim_feedforward = 256

    # Initialize model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    model = MultiModalTransformer(
        video_dim=video_dim,
        audio_dim=audio_dim,
        n_heads_video=n_heads_video,
        n_heads_audio=n_heads_audio,
        n_heads_transformer=n_heads_transformer,
        num_layers=num_layers,
        d_model=d_model,
        dim_feedforward=dim_feedforward,
        output_dim=1,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5
    )
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3
    )
    best_val_loss = float("inf")

    # Define the attention mask once (assuming fixed seq_len=225)
    seq_len = 225
    attn_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(device)

    # Training Loop
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        model.train()
        total_loss = 0.0
        total_primary_loss = 0.0
        total_reg_loss = 0.0

        for batch in train_loader:
            video_features, audio_features, actual_length, y_offset = batch

            video_features = video_features.to(device)
            audio_features = audio_features.to(device)
            y_offset = y_offset.to(device).float()

            # Create key_padding_mask based on actual sequence lengths
            # Initialize mask as False (no padding)
            key_padding_mask = (
                torch.zeros(video_features.size(0), seq_len).bool().to(device)
            )
            for i, length in enumerate(actual_length):
                if length < seq_len:
                    key_padding_mask[i, length:] = True  # True indicates padding

            optimizer.zero_grad()

            # Forward pass with both masks
            output = (
                model(
                    video_features,
                    audio_features,
                    mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                )
                .view(-1)
                .float()
            )

            # Compute losses
            primary_loss = criterion(output, y_offset)
            reg_loss = variance_regularization(
                output, alpha=alphas[epoch]
            )  # Adjust alpha as needed
            loss = primary_loss + reg_loss

            # Accumulate losses for logging
            total_loss += loss.item()
            total_primary_loss += primary_loss.item()
            total_reg_loss += reg_loss.item()

            # Backpropagation
            loss.backward()

            # Optional: Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

        # Calculate average losses
        avg_loss = total_loss / len(train_loader)
        avg_primary_loss = total_primary_loss / len(train_loader)
        avg_reg_loss = total_reg_loss / len(train_loader)

        print(
            f"\nEpoch {epoch + 1}/{num_epochs}, Total Loss: {avg_loss:.4f}, "
            f"Primary Loss: {avg_primary_loss:.4f}, Reg Loss: {avg_reg_loss:.4f}",
            f"RMSE: {avg_loss**0.5:.4f}",
            flush=True,
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

                # Create key_padding_mask for validation
                key_padding_mask = (
                    torch.zeros(video_features.size(0), seq_len).bool().to(device)
                )
                for i, length in enumerate(actual_length):
                    if length < seq_len:
                        key_padding_mask[i, length:] = True  # True indicates padding

                # Forward pass with both masks
                outputs = (
                    model(
                        video_features,
                        audio_features,
                        mask=attn_mask,
                        key_padding_mask=key_padding_mask,
                    )
                    .view(-1)
                    .float()
                )

                # Compute loss
                loss = criterion(outputs, y_offset)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}, RMSE: {avg_val_loss**0.5:.4f}")
        scheduler.step(avg_val_loss)

        # writer.add_scalar("Loss/Train", avg_primary_loss, epoch)
        # writer.add_scalar("Loss/Validation", avg_val_loss, epoch)

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(
                f"Saved best model with validation loss: {best_val_loss:.4f}, RMSE: {best_val_loss**0.5:.4f}",
                flush=True,
            )

    # Save the final model
    torch.save(model.state_dict(), final_model_path)
    print("Successfully finished training!")
