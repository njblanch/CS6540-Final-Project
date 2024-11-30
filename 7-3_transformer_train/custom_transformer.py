# custom_transformer.py

import torch
import torch.nn as nn
import math


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256, dropout=0.1):
        super(LearnablePositionalEncoding, self).__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)
        nn.init.xavier_uniform_(self.pos_embedding.weight)

    def forward(self, x, positions=None):
        if positions is None:
            batch_size, seq_len, _ = x.size()
            positions = (
                torch.arange(0, seq_len, device=x.device)
                .unsqueeze(0)
                .expand(batch_size, seq_len)
            )
        pos_embed = self.pos_embedding(positions)
        return self.dropout(x + pos_embed)


class GatedFusion(nn.Module):
    """
    Gated fusion mechanism to combine audio and video features.
    """

    def __init__(self, video_dim, audio_dim, d_model, gate_activation="sigmoid"):
        super(GatedFusion, self).__init__()
        self.fc = nn.Linear(video_dim + audio_dim, d_model)
        if gate_activation == "sigmoid":
            self.gate = nn.Sequential(
                nn.Linear(video_dim + audio_dim, d_model), nn.Sigmoid()
            )
        elif gate_activation == "softmax":
            self.gate = nn.Sequential(
                nn.Linear(video_dim + audio_dim, d_model), nn.Softmax(dim=-1)
            )
        elif gate_activation == "tanh":
            self.gate = nn.Sequential(
                nn.Linear(video_dim + audio_dim, d_model), nn.Tanh()
            )
        else:
            raise ValueError(f"Unsupported gate_activation: {gate_activation}")

    def forward(self, video_features, audio_features):
        """
        Forward pass for gated fusion.

        Args:
            video_features (Tensor): Tensor of shape (batch_size, seq_len, video_dim)
            audio_features (Tensor): Tensor of shape (batch_size, seq_len, audio_dim)

        Returns:
            Tensor: Fused features of shape (batch_size, seq_len, d_model)
        """
        combined = torch.cat((video_features, audio_features), dim=-1)
        fused = self.fc(combined)
        gate = self.gate(combined)
        return fused * gate


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention layer where one modality can attend to the other.
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super(CrossModalAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """
        Args:
            query, key, value: Tensors of shape (batch_size, seq_len, d_model)
            attn_mask: Optional tensor for masking attention
            key_padding_mask: Optional tensor for padding
        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        attn_output, _ = self.multihead_attn(
            query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )
        attn_output = self.dropout(attn_output)
        return self.layer_norm(query + attn_output)


class MultiModalTransformer(nn.Module):
    def __init__(
        self,
        video_dim,
        audio_dim,
        n_heads_transformer,
        num_layers,
        d_model,
        dim_feedforward,
        output_dim,
        max_seq_length=256,
        dropout=0.1,
        max_offset=15,
    ):
        super(MultiModalTransformer, self).__init__()
        self.max_offset = max_offset  # +- max frame offset for regression

        if d_model % n_heads_transformer != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads_transformer ({n_heads_transformer})"
            )

        # Projection layers
        self.video_proj = nn.Linear(video_dim, d_model)
        self.audio_proj = nn.Linear(audio_dim, d_model)

        # Shared positional encoding
        self.pos_encoder = LearnablePositionalEncoding(d_model, max_len=max_seq_length)

        # Cross-modal attention layers
        self.cross_attn_video_to_audio = CrossModalAttention(
            d_model, n_heads_transformer, dropout=dropout
        )
        self.cross_attn_audio_to_video = CrossModalAttention(
            d_model, n_heads_transformer, dropout=dropout
        )

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads_transformer,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # [CLS] token for pooling
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Output layer for regression
        self.output_layer = nn.Linear(d_model, output_dim)
        self.output_activation = nn.Tanh()

        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout)

        # Initialize parameters
        self._init_weights()

    def _init_weights(self):
        # Initialize weights using Xavier uniform for linear layers and embeddings
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.MultiheadAttention):
                # Initialize MultiheadAttention
                nn.init.xavier_uniform_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.constant_(module.in_proj_bias, 0)
                if module.out_proj.weight is not None:
                    nn.init.xavier_uniform_(module.out_proj.weight)
                if module.out_proj.bias is not None:
                    nn.init.constant_(module.out_proj.bias, 0)

    def forward(
        self,
        video_features,
        audio_features,
        video_positions,
        audio_positions,
        mask=None,
        key_padding_mask=None,
    ):
        # Project features to the common dimension
        video_features = self.video_proj(video_features)  # (batch, seq, d_model)
        audio_features = self.audio_proj(audio_features)  # (batch, seq, d_model)

        # Add positional encoding
        video_features = self.pos_encoder(video_features, positions=video_positions)
        audio_features = self.pos_encoder(audio_features, positions=audio_positions)

        # Cross-modal Attention
        fused_features_video_to_audio = self.cross_attn_video_to_audio(
            query=video_features,
            key=audio_features,
            value=audio_features,
            attn_mask=mask,
            key_padding_mask=key_padding_mask,
        )

        fused_features_audio_to_video = self.cross_attn_audio_to_video(
            query=audio_features,
            key=video_features,
            value=video_features,
            attn_mask=mask,
            key_padding_mask=key_padding_mask,
        )

        # Combine both cross-attended features
        fused_features = (
            fused_features_video_to_audio + fused_features_audio_to_video
        ) / 2

        # Add [CLS] token
        batch_size, seq_len, _ = fused_features.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
        fused_features = torch.cat(
            (cls_tokens, fused_features), dim=1
        )  # (batch, seq + 1, d_model)

        # Adjust key_padding_mask for Transformer Encoder
        if key_padding_mask is not None:
            cls_mask = torch.zeros(
                (key_padding_mask.size(0), 1),
                dtype=key_padding_mask.dtype,
                device=key_padding_mask.device,
            )
            transformer_key_padding_mask = torch.cat(
                [cls_mask, key_padding_mask], dim=1
            )  # (batch, seq + 1)
        else:
            transformer_key_padding_mask = None

        # Pass through Transformer Encoder
        transformer_out = self.transformer_encoder(
            fused_features, src_key_padding_mask=transformer_key_padding_mask
        )  # (batch, seq + 1, d_model)

        # Extract [CLS] token representation
        cls_output = transformer_out[:, 0, :]  # (batch, d_model)

        # Apply dropout
        cls_output = self.dropout(cls_output)

        # Final regression output
        output = self.output_layer(cls_output)  # (batch, output_dim)
        output = self.output_activation(output) * self.max_offset

        return output
