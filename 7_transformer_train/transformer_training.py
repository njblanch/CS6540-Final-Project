from custom_transformer import DualInputTransformer, MultimodalTransformer
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

    train_data, val_data, test_data = load_dataset(video_path, audio_path, save=True, max_data=None)# {"train": 50, "test": 10, "val": 10}) # Save train test split
    print("Shapes!", flush=True)
    print(len(train_data))
    print(len(val_data))
    print(len(test_data))
    # print(train_data[0].shape)
    # print(val_data[0].shape)
    # print(test_data[0].shape, flush=True)

    # Hyperparameters
    num_epochs = 100
    batch_size = 32
    learning_rate = 1e-5

    audio_dim = 118
    video_dim = 1280
    n_heads = 6 # Needs to be d_model/n_heads is an int
    num_layers = 6 # Change as needed
    d_model = audio_dim + video_dim # audio_dim + video_dim
    dim_feedforward = 256 # Change as needed

    # Initialize model, optimizer, and loss function
    # model = DualInputTransformer(audio_dim, video_dim, n_heads, num_layers, d_model, dim_feedforward)
    model = MultimodalTransformer(features_dim=d_model, nhead=n_heads, num_layers=num_layers, d_model=d_model,
                                  output_dim=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("after loading")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for features, attention_mask, target in train_data:
            optimizer.zero_grad()
            features.to(device)
            attention_mask.to(device)
            target.to(device)

            # Forward pass, using the attention mask to ignore padding
            # output = model(features, src_key_padding_mask=attention_mask)
            output = model(features, src_mask=attention_mask)
            # Compute loss and backpropagate

            loss = criterion(output.view(-1).float(), target.view(-1).float())
            # print(output)
            # print(target, flush=True)
            total_loss += loss.item()

            loss.float()

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
        # print(total_loss)
        avg_loss = total_loss / len(train_data)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}", flush=True)

        # Validation Loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for features, attention_mask, target in val_data:
                # Forward pass, using the attention mask to ignore padding
                features.to(device)
                attention_mask.to(device)
                target.to(device)
                # outputs = model(features, src_key_padding_mask=attention_mask)
                outputs = model(features, src_mask=attention_mask)

                # Compute the loss
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
