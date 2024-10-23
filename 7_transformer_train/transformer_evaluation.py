from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from daul_input_transformer import DualInputTransformer
from data_loading import VideoAudioDataset


def evaluate_model(model, test_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():  # Disable gradient calculations
        for audio_features, video_features, target in test_loader:
            outputs = model(audio_features, video_features)
            loss = criterion(outputs.view(-1), target.view(-1))
            total_loss += loss.item()

            all_preds.extend(outputs.cpu().numpy())  # Store predictions
            all_targets.extend(target.cpu().numpy())  # Store targets

    avg_test_loss = total_loss / len(test_loader)
    mse = mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    return avg_test_loss, mse, mae


def read_filenames(filename):
    with open(filename, 'r') as file:
        filenames = [line.strip() for line in file]
    return filenames


if __name__=="__main__":
    # Load model - Should mirror the training
    audio_dim = 50
    video_dim = 1280
    n_heads = 8
    num_layers = 2
    d_model = 100
    dim_feedforward = 100

    # Load data
    audio_path = "/gpfs2/classes/cs6540/AVSpeech/5_audio/train"
    video_path = "/gpfs2/classes/cs6540/AVSpeech/6_visual_features/train_dist"
    test_filenames = read_filenames("test_filenames.txt")
    test_dataset = VideoAudioDataset(test_filenames, video_path, audio_path)

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # Initialize model, optimizer, and loss function
    model = DualInputTransformer(audio_dim, video_dim, n_heads, num_layers, d_model, dim_feedforward)
    model.load_state_dict(torch.load('transformer_desync_model.pth'))

    criterion = nn.MSELoss()

    avg_test_loss, mse, mae = evaluate_model(model, test_loader, criterion)
    print(f"Validation Loss: {avg_test_loss:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")
