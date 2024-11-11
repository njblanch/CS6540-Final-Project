# improved_transformer.py

import torch
import torch.nn as nn
import math


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional embeddings for sequences.
    """

    def __init__(self, d_model, max_len=5000):
        super(LearnablePositionalEncoding, self).__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.max_len = max_len

        # Initialize positional embeddings
        nn.init.normal_(self.pos_embedding.weight, mean=0, std=0.1)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor with positional embeddings added.
        """
        batch_size, seq_len, _ = x.size()
        positions = (
            torch.arange(0, seq_len, device=x.device)
            .unsqueeze(0)
            .expand(batch_size, seq_len)
        )
        pos_embed = self.pos_embedding(positions)  # (batch_size, seq_len, d_model)
        return x + pos_embed


class GatedFusion(nn.Module):
    """
    Gated fusion mechanism to combine audio and video features.
    """

    def __init__(self, video_dim, audio_dim, d_model):
        super(GatedFusion, self).__init__()
        self.fc = nn.Linear(video_dim + audio_dim, d_model)
        self.gate = nn.Sequential(
            nn.Linear(video_dim + audio_dim, d_model), nn.Sigmoid()
        )

    def forward(self, video_features, audio_features):
        """
        Args:
            video_features: Tensor of shape (batch_size, seq_len, video_dim)
            audio_features: Tensor of shape (batch_size, seq_len, audio_dim)
        Returns:
            Fused features: Tensor of shape (batch_size, seq_len, d_model)
        """
        combined = torch.cat(
            (video_features, audio_features), dim=-1
        )  # (batch, seq, video_dim + audio_dim)
        fused = self.fc(combined)  # (batch, seq, d_model)
        gate = self.gate(combined)  # (batch, seq, d_model)
        return fused * gate  # (batch, seq, d_model)


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
        n_heads_video,
        n_heads_audio,
        n_heads_transformer,
        num_layers,
        d_model,
        dim_feedforward,
        output_dim,
        max_seq_length=5000,
        dropout=0.1,
    ):
        super(MultiModalTransformer, self).__init__()

        # Ensure dimensions are divisible by the number of heads
        assert (
            video_dim % n_heads_video == 0
        ), "video_dim must be divisible by n_heads_video"
        assert (
            audio_dim % n_heads_audio == 0
        ), "audio_dim must be divisible by n_heads_audio"
        assert (
            d_model % n_heads_transformer == 0
        ), "d_model must be divisible by n_heads_transformer"

        # Positional encodings for video and audio (learnable)
        self.pos_encoder_video = LearnablePositionalEncoding(
            video_dim, max_len=max_seq_length
        )
        self.pos_encoder_audio = LearnablePositionalEncoding(
            audio_dim, max_len=max_seq_length
        )

        # Separate multi-head attention layers for video and audio
        self.video_attention = nn.MultiheadAttention(
            embed_dim=video_dim,
            num_heads=n_heads_video,
            dropout=dropout,
            batch_first=True,
        )
        self.audio_attention = nn.MultiheadAttention(
            embed_dim=audio_dim,
            num_heads=n_heads_audio,
            dropout=dropout,
            batch_first=True,
        )

        # Gated Fusion layer
        self.gated_fusion = GatedFusion(video_dim, audio_dim, d_model)

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

        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout)

        # Initialize parameters
        self._init_weights()

    def _init_weights(self):
        # Initialize weights using Xavier uniform for linear layers and normal for embeddings
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)
            elif isinstance(module, nn.MultiheadAttention):
                # Initialize MultiheadAttention
                nn.init.xavier_uniform_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.constant_(module.in_proj_bias, 0)
                if module.out_proj.weight is not None:
                    nn.init.xavier_uniform_(module.out_proj.weight)
                if module.out_proj.bias is not None:
                    nn.init.constant_(module.out_proj.bias, 0)

    def forward(self, video_features, audio_features, mask=None, key_padding_mask=None):
        """
        Args:
            video_features: (batch_size, seq_len, video_dim)
            audio_features: (batch_size, seq_len, audio_dim)
            mask: (seq_len, seq_len) - attn_mask (optional)
            key_padding_mask: (batch_size, seq_len) - mask for padded tokens (optional)
        Returns:
            output: (batch_size, output_dim)
        """
        # Add positional encoding
        video_features = self.pos_encoder_video(
            video_features
        )  # (batch, seq, video_dim)
        audio_features = self.pos_encoder_audio(
            audio_features
        )  # (batch, seq, audio_dim)

        # Apply multi-head attention separately to video and audio features
        video_out, _ = self.video_attention(
            video_features,
            video_features,
            video_features,
            attn_mask=mask,
            key_padding_mask=key_padding_mask,
        )
        audio_out, _ = self.audio_attention(
            audio_features,
            audio_features,
            audio_features,
            attn_mask=mask,
            key_padding_mask=key_padding_mask,
        )

        # Gated Fusion
        fused_features = self.gated_fusion(
            video_out, audio_out
        )  # (batch, seq, d_model)

        # Cross-modal Attention
        fused_features = self.cross_attn_video_to_audio(
            fused_features,
            fused_features,
            fused_features,
            mask=mask,
            key_padding_mask=key_padding_mask,
        )
        fused_features = self.cross_attn_audio_to_video(
            fused_features,
            fused_features,
            fused_features,
            mask=mask,
            key_padding_mask=key_padding_mask,
        )

        # Add [CLS] token
        batch_size, seq_len, _ = fused_features.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
        fused_features = torch.cat(
            (cls_tokens, fused_features), dim=1
        )  # (batch, seq + 1, d_model)

        # Pass through Transformer Encoder
        transformer_out = self.transformer_encoder(
            fused_features
        )  # (batch, seq + 1, d_model)

        # Extract [CLS] token representation
        cls_output = transformer_out[:, 0, :]  # (batch, d_model)

        # Apply dropout
        cls_output = self.dropout(cls_output)

        # Final regression output
        output = self.output_layer(cls_output)  # (batch, output_dim)

        return output
