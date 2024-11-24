import torch
import torch.nn as nn



def _compute_flattened_size(stream, input_size):
    """
    Helper function to compute the size of the flattened output after convolutional layers.
    :param stream: Sequential layers of the stream.
    :param input_size: Tuple of input dimensions (Channels, Temporal, Height, Width or Features).
    :return: Flattened size as an integer.
    """
    with torch.no_grad():
        dummy_input = torch.zeros(1, *input_size)  # Batch size = 1
        output = stream(dummy_input)
        return output.numel()


class AudioVideoLSTM(nn.Module):
    def __init__(self, audio_feature_dim=(1, 60, 118), video_feature_dim=(16, 60, 8, 8), lstm_hidden_dim=128, output_dim=1, num_layers=2, dropout=0.3):
        super(AudioVideoLSTM, self).__init__()

        self.audio_input_size = audio_feature_dim
        self.visual_input_size = video_feature_dim

        # Visual stream
        self.visual_stream = nn.Sequential(
            # 3D Convolution + ReLU + MaxPooling + Dropout
            nn.Conv3d(16, 64, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2),
            nn.Dropout3d(p=0.3),

            # 3D Conv + BatchNorm + ReLU + MaxPooling + Dropout
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2),
            nn.Dropout3d(p=0.3),

            # Flatten per time step (across spatial dimensions)
            nn.Flatten(start_dim=2)  # Flatten height and width for each time step
        )

        # Dynamically compute the flattened size for FC layers
        # self.visual_fc_input_size = _compute_flattened_size(self.visual_stream, (16, 60, 8, 8))

        # self.visual_fc = nn.Sequential(
        #     nn.Linear(self.visual_fc_input_size, 1024),
        #     nn.ReLU()
        # )

        self.audio_stream = nn.Sequential(
            # First 2D Convolution (only temporal comparison, width of kernel is 1)
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=1, padding=(1, 1)),  # Only temporal convolution, height
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # Temporal pooling only, stride=1 to maintain time steps

            # Second 2D Convolution (enable cross-feature comparison)
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # Pooling both temporal and feature dimensions

            # Flatten per time step (across feature dimensions)
            nn.Flatten(start_dim=2)  # Flatten height (features) for each time step
        )


        # Compute the flattened size for the fully connected layer after the 2D convolutions
        # self.audio_fc_input_size = _compute_flattened_size(self.audio_stream, (1, 60, 118))

        # self.audio_fc = nn.Sequential(
        #     nn.Linear(self.audio_fc_input_size, 1024),
        #     nn.ReLU()
        # )

        # LSTM layer for audio
        self.audio_lstm = nn.LSTM(input_size=435, hidden_size=lstm_hidden_dim, 
                                  num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        
        # LSTM layer for vide
        self.video_lstm = nn.LSTM(input_size=60, hidden_size=lstm_hidden_dim, 
                                  num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=True)

        # Fully connected layer for combining audio and video outputs
        self.fc = nn.Linear(lstm_hidden_dim * 4, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, audio_input, visual_input):
        # Visual stream
        # Reshape flattened visual input into (Batch, Channels, Temporal, Height, Width)
        batch_size, num_frames, flattened_features = visual_input.size()
        height, width, depth = 8, 8, 16  # Known dimensions
        # print(visual_input.shape)
        visual_input = visual_input.view(batch_size, num_frames, height, width, depth)
        visual_input = visual_input.permute(0, 4, 1, 2, 3)
        # print("visual_input shape before audio_stream:", visual_input.shape, flush=True)
        visual_features = self.visual_stream(visual_input)
        # print("visual_features shape after conv + flatten:", visual_features.shape, flush=True)

        # Audio stream
        audio_input = audio_input.unsqueeze(1)  # Add channel dimension [Batch, Channels=1, Temporal, Features]
        # print(audio_input.shape)
        # print("audio_input shape before audio_stream:", audio_input.shape, flush=True)
        audio_features = self.audio_stream(audio_input)
        # print("audio_features shape after conv + flatten:", audio_features.shape, flush=True)

        _, (audio_hidden, _) = self.audio_lstm(audio_features)
        # print("audio_hidden shape:", audio_hidden.shape, flush=True)
        audio_out = torch.cat((audio_hidden[-2], audio_hidden[-1]), dim=1)  # Use the last layer's hidden state for prediction

        _, (video_hidden, _) = self.video_lstm(visual_features)
        video_out = torch.cat((video_hidden[-2], video_hidden[-1]), dim=1)  # Use the last layer's hidden state for prediction

        # Concatenate the final hidden states from audio and video
        combined_features = torch.cat((audio_out, video_out), dim=1)
        combined_features = self.dropout(combined_features)

        # Pass through fully connected layer to get the output (one-hot encoded class probabilities for each offset)
        output = self.fc(combined_features)

        return output
