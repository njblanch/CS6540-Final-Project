from dual_input_transformer import DualInputTransformer
from torch.utils.data import DataLoader
import torch
import torch.nn as nn


if __name__ == "__main__":
    # I would recommend we use a validation set too - this helps with overfitting and tracking the model's overfitting
    # a lot easier. This logic is added in the training loop, saves models of best loss
    final_model_path = "transformer_desync_model_final.pth"
    best_model_path = "transformer_desync_model_best.pth"

    train_loader = ...  # Add data loading
    val_loader = ... # Add data loading

    # Hyperparameters
    num_epochs = 10
    batch_size = 32
    learning_rate = 1e-4

    audio_dim = 50 # TODO: Set this
    video_dim = 1280 # TODO: Verify
    n_heads = 8 # Change as needed
    num_layers = 2 # Change as needed
    d_model = 100 # Change as needed
    dim_feedforward = 100 # Change as needed

    # Initialize model, optimizer, and loss function
    model = DualInputTransformer(audio_dim, video_dim, n_heads, num_layers, d_model, dim_feedforward)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for audio_features, video_features, target in train_loader:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(audio_features, video_features)

            # Compute loss
            loss = criterion(outputs.view(-1), target.view(-1))
            total_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

        # Validation Loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for audio_features, video_features, target in val_loader:
                outputs = model(audio_features, video_features)
                loss = criterion(outputs.view(-1), target.view(-1))
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")

    # TODO: only saving after completion, use of a validation set could be beneficial
    torch.save(model.state_dict(), final_model_path)



