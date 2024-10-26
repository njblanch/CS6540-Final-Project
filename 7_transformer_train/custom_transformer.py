import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class DualInputTransformer(nn.Module):
    def __init__(self, audio_dim, video_dim, n_heads, num_layers, d_model, dim_feedforward):
        super(DualInputTransformer, self).__init__()
        self.combined_embedding = nn.Linear(d_model, d_model)

        encoder_layers = TransformerEncoderLayer(d_model, n_heads, dim_feedforward, activation="gelu", dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.fc_out = nn.Linear(d_model, 1)  # Output for predicting desync frames

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, features, src_key_padding_mask):
        # print(type(features))
        # print(features)
        combined = self.combined_embedding(features)
        # print(type(combined))
        # print(combined)
        # Combine the features
        assert not torch.isnan(combined).any(), "Input contains NaN values"
        # print(f"Input stats - min: {features.min()}, max: {features.max()}, mean: {features.mean()}")
        # transformer_output = self.transformer_encoder(combined, src_key_padding_mask=src_key_padding_mask.T)
        transformer_output, layer_outputs = self.transformer_encoder(combined,
                                                                     mask=src_key_padding_mask.T)
        # print(type(transformer_output))
        # print(transformer_output.dtype)
        # print(transformer_output)

        # Output layer
        return self.fc_out(transformer_output.mean(dim=1)), layer_outputs


class MultimodalTransformer(nn.Module):
    def __init__(self, features_dim, d_model, nhead, num_layers, output_dim):
        super(MultimodalTransformer, self).__init__()

        # Audio feature processing
        self.fc_in = nn.Linear(features_dim, d_model)

        # Transformer
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers)

        # Final prediction layer
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, features, src_mask=None):
        # Process audio features
        # print(features.shape)
        features = features.permute(1, 0, 2)
        # print(features.shape)

        assert not torch.isnan(features).any(), "Input contains NaN values"
        features_emb = self.fc_in(features)

        # Pass through transformer
        transformer_out = self.transformer(features_emb, features_emb, src_key_padding_mask=src_mask)  # Shape: (2, batch_size, d_model)

        # Take output from the last time step (video in this case)
        final_out = transformer_out[-1]  # Shape: (batch_size, d_model)

        # Final prediction
        output = self.fc(final_out)  # Shape: (batch_size, output_dim)

        return output


if __name__ == "__main__":
    # Define parameters
    sequence_length = 10  # For example, a sequence of 10 time steps
    batch_size = 10  # You can adjust this as needed
    feature_dimension = 118 + 1280  # Example: audio_dim + video_dim
    audio_input_dim = 118
    video_input_dim = 1280

    # Generate random input tensor
    random_input = torch.rand(sequence_length, batch_size, feature_dimension)  # Values in [0, 1]

    # Optionally, scale to [-1, 1]
    random_input = 2 * random_input - 1

    # attention_mask = torch.ones(sequence_length, batch_size, dtype=torch.bool)  # Shape: (sequence_length, batch_size)

    model = MultimodalTransformer(features_dim=1398, nhead=6, num_layers=1, d_model=1398, output_dim=1)

    # Example inputs
    features = torch.rand(sequence_length, batch_size, feature_dimension)

    # Example attention mask (if using padding, shape should be (batch_size, 2))
    # Assume both audio and video features are of the same length for simplicity
    attention_mask = torch.ones(batch_size, sequence_length, dtype=torch.bool)

    output = model(features, src_mask=attention_mask)
    print(output.shape)
    print(output)
