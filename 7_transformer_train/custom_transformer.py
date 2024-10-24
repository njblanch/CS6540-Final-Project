import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class DualInputTransformer(nn.Module):
    def __init__(self, audio_dim, video_dim, n_heads, num_layers, d_model, dim_feedforward):
        super(DualInputTransformer, self).__init__()
        self.audio_embedding = nn.Linear(audio_dim, d_model)
        self.video_embedding = nn.Linear(video_dim, d_model)

        encoder_layers = TransformerEncoderLayer(d_model, n_heads, dim_feedforward)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.fc_out = nn.Linear(d_model, 1)  # Output for predicting desync frames

    def forward(self, features):
        combined = self.audio_embedding(features)

        # Combine the features
        transformer_output = self.transformer_encoder(combined)

        # Output layer
        return self.fc_out(transformer_output.mean(dim=1))

