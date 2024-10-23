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

    def forward(self, audio_features, video_features):
        audio_embedded = self.audio_embedding(audio_features)
        video_embedded = self.video_embedding(video_features)

        # Combine the features
        combined_features = audio_embedded + video_embedded
        transformer_output = self.transformer_encoder(combined_features)

        # Output layer
        return self.fc_out(transformer_output.mean(dim=1))

