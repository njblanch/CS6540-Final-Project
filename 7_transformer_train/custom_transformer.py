import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F


class MultimodalTransformer(nn.Module):
    def __init__(self, features_dim, d_model, nhead, num_layers, output_dim, device):
        super(MultimodalTransformer, self).__init__()

        # Audio feature processing
        self.fc_in = nn.Linear(features_dim, d_model)
        # self.fc_in.to(device)

        # Transformer
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers)
        # self.transformer.to(device)

        # Final prediction layer
        self.fc = nn.Linear(d_model, output_dim)
        # self.fc.to(device)
        #
        # self.device = device
        # self.to(device)

    def forward(self, features, src_mask=None):
        # Process audio features
        # print(features.shape)
        features = features.permute(1, 0, 2)
        # print(features.shape)
        # features.to(self.device)


        assert not torch.isnan(features).any(), "Input contains NaN values"
        features_emb = self.fc_in(features)

        # Pass through transformer
        transformer_out = self.transformer(features_emb, features_emb, src_key_padding_mask=src_mask)  # Shape: (2, batch_size, d_model)

        # Take output from the last time step (video in this case)
        final_out = transformer_out[-1]  # Shape: (batch_size, d_model)

        # Final prediction
        output = self.fc(final_out)  # Shape: (batch_size, output_dim)

        return output



