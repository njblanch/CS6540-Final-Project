import torch
import torch.nn as nn
import torch.nn.functional as F


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


class AVsync_CNN(nn.Module):
    def __init__(self, audio_dim=(1, 225, 118), video_dim=(16, 225, 8, 8), time_steps=225):
        super(AVsync_CNN, self).__init__()

        self.audio_input_size = audio_dim
        self.visual_input_size = video_dim

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
        )

        # Dynamically compute the flattened size for FC layers
        self.visual_fc_input_size = _compute_flattened_size(self.visual_stream, (16, 225, 8, 8))

        self.visual_fc = nn.Sequential(
            nn.Linear(self.visual_fc_input_size, 1024),
            nn.ReLU()
        )

        self.audio_stream = nn.Sequential(
            # First 2D Convolution (only temporal comparison, width of kernel is 1)
            # nn.Conv2d(1, 64, kernel_size=(3, 1), stride=1, padding=(1, 0)),  # Only temporal convolution, height
            #             nn.ReLU(),
            #             nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # Temporal pooling only
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=1, padding=(1, 1)),  # Only temporal convolution, height
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # Temporal pooling only

            # Second 2D Convolution (enable cross-feature comparison)
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # Pooling both temporal and feature dimensions
        )

        # Compute the flattened size for the fully connected layer after the 2D convolutions
        self.audio_fc_input_size = _compute_flattened_size(self.audio_stream, (1, 225, 118))

        self.audio_fc = nn.Sequential(
            nn.Linear(self.audio_fc_input_size, 1024),
            nn.ReLU()
        )

        # Fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(1024 + 1024, 1024),  # Combine visual and audio streams
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)  # Output: temporal offset
        )

    def forward(self, visual_input, audio_input, visual_mask, audio_mask):
        # Visual stream
        # Reshape flattened visual input into (Batch, Channels, Temporal, Height, Width)
        batch_size, num_frames, flattened_features = visual_input.size()
        height, width, depth = 8, 8, 16  # Known dimensions
        # print(visual_input.shape)
        visual_input = visual_input.view(batch_size, num_frames, height, width, depth)
        visual_input = visual_input.permute(0, 4, 1, 2, 3)  # Change to (Batch, Channels, Temporal, Height, Width)
        # (16, 225, 8, 8)
        # print(visual_input.shape)
        # Visual stream
        visual_features = self.visual_stream(visual_input)
        visual_features = visual_features.reshape(visual_features.size(0), -1)  # Flatten

        # Apply attention mask to the visual features
        # TODO: Add masking, will have to modify mask to match
        # visual_mask = visual_mask.unsqueeze(1)  # Shape [Batch, Temporal, 1]
        # visual_features = visual_features * visual_mask.reshape(batch_size, -1)  # Masking

        visual_features = self.visual_fc(visual_features)

        # Audio stream
        audio_input = audio_input.unsqueeze(1)  # Add channel dimension [Batch, Channels=1, Temporal, Features]
        # print(audio_input.shape)
        audio_features = self.audio_stream(audio_input)
        audio_features = audio_features.view(audio_features.size(0), -1)  # Flatten
        audio_features = self.audio_fc(audio_features)

        # Apply attention mask to the audio features
        # TODO: Add masking, modify to match sizing of outputs
        # audio_mask = audio_mask.unsqueeze(1)  # Shape [Batch, Temporal, 1]
        # audio_features = audio_features * audio_mask.view(batch_size, -1)  # Masking

        # Fusion
        fused_features = torch.cat((visual_features, audio_features), dim=1)
        output = self.fusion_network(fused_features)

        return output



class AVsync_CNN_MFCC(nn.Module):
    def __init__(self, audio_dim=(1, 225, 12), video_dim=(16, 225, 8, 8), time_steps=225):
        super(AVsync_CNN_MFCC, self).__init__()

        self.audio_input_size = audio_dim
        self.visual_input_size = video_dim

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
        )

        # Dynamically compute the flattened size for FC layers
        self.visual_fc_input_size = _compute_flattened_size(self.visual_stream, (16, 225, 8, 8))

        self.visual_fc = nn.Sequential(
            nn.Linear(self.visual_fc_input_size, 1024),
            nn.ReLU()
        )

        self.audio_stream = nn.Sequential(
            # First 2D Convolution (only temporal comparison, width of kernel is 1)
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=1, padding=(1, 1)),  # Only temporal convolution, height
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # Temporal pooling only

            # Second 2D Convolution (enable cross-feature comparison)
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # Pooling both temporal and feature dimensions
        )

        # Compute the flattened size for the fully connected layer after the 2D convolutions
        self.audio_fc_input_size = _compute_flattened_size(self.audio_stream, (1, 225, 12))

        self.audio_fc = nn.Sequential(
            nn.Linear(self.audio_fc_input_size, 1024),
            nn.ReLU()
        )

        # Fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(1024 + 1024, 1024),  # Combine visual and audio streams
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)  # Output: temporal offset
        )

    def forward(self, visual_input, audio_input, visual_mask, audio_mask):
        # Visual stream
        # Reshape flattened visual input into (Batch, Channels, Temporal, Height, Width)
        batch_size, num_frames, flattened_features = visual_input.size()
        height, width, depth = 8, 8, 16  # Known dimensions
        # print(visual_input.shape)
        visual_input = visual_input.view(batch_size, num_frames, height, width, depth)
        visual_input = visual_input.permute(0, 4, 1, 2, 3)  # Change to (Batch, Channels, Temporal, Height, Width)
        # (16, 225, 8, 8)
        # print(visual_input.shape)
        # Visual stream
        visual_features = self.visual_stream(visual_input)
        visual_features = visual_features.reshape(visual_features.size(0), -1)  # Flatten

        # Apply attention mask to the visual features
        # TODO: Add masking, will have to modify mask to match
        # visual_mask = visual_mask.unsqueeze(1)  # Shape [Batch, Temporal, 1]
        # visual_features = visual_features * visual_mask.reshape(batch_size, -1)  # Masking

        visual_features = self.visual_fc(visual_features)

        # Audio stream
        audio_input = audio_input.unsqueeze(1)  # Add channel dimension [Batch, Channels=1, Temporal, Features]
        # print(audio_input.shape)
        audio_features = self.audio_stream(audio_input)
        audio_features = audio_features.view(audio_features.size(0), -1)  # Flatten
        audio_features = self.audio_fc(audio_features)

        # Apply attention mask to the audio features
        # TODO: Add masking, modify to match sizing of outputs
        # audio_mask = audio_mask.unsqueeze(1)  # Shape [Batch, Temporal, 1]
        # audio_features = audio_features * audio_mask.view(batch_size, -1)  # Masking

        # Fusion
        fused_features = torch.cat((visual_features, audio_features), dim=1)
        output = self.fusion_network(fused_features)

        return output
