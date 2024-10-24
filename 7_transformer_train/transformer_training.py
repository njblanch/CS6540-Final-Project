from custom_transformer import DualInputTransformer
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from data_loading import load_dataset

if __name__ == "__main__":
    # I would recommend we use a validation set too - this helps with overfitting and tracking the model's overfitting
    # a lot easier. This logic is added in the training loop, saves models of best loss
    final_model_path = "transformer_desync_model_final.pth"
    best_model_path = "transformer_desync_model_best.pth"

    audio_path = "/gpfs2/classes/cs6540/AVSpeech/5_audio/train"
    video_path = "/gpfs2/classes/cs6540/AVSpeech/6_visual_features/train_dist"

    train_data, val_data, test_data = load_dataset(video_path, audio_path, save=True, max_data = {"train": 50, "test": 10, "val": 10}) # Save train test split
    print("Shapes!", flush=True)
    print(len(train_data))
    print(len(val_data))
    print(len(test_data))
    # print(train_data[0].shape)
    # print(val_data[0].shape)
    # print(test_data[0].shape, flush=True)

    # Hyperparameters
    num_epochs = 10
    batch_size = 32
    learning_rate = 1e-4

    audio_dim = 119 # TODO: Set this
    video_dim = 1280
    n_heads = 8 # Needs to be d_model/n_heads is an int
    num_layers = 2 # Change as needed
    d_model = 80 # audio_dim + video_dim
    dim_feedforward = 100 # Change as needed

    # Initialize model, optimizer, and loss function
    model = DualInputTransformer(audio_dim, video_dim, n_heads, num_layers, d_model, dim_feedforward)
    print("after loading")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for features, target in train_data:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(features)

            # Compute loss
            loss = criterion(outputs.view(-1), target.view(-1))
            total_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_data)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}", flush=True)

        # Validation Loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for features, target in val_data:
                outputs = model(features)
                loss = criterion(outputs.view(-1), target.view(-1))
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_data)
        print(f"Validation Loss: {avg_val_loss:.4f}", flush=True)

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with validation loss: {best_val_loss:.4f}", flush=True)

    # TODO: only saving after completion, use of a validation set could be beneficial
    torch.save(model.state_dict(), final_model_path)

    print("Successfully finished training!")
