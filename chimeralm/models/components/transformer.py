import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """Sine-cosine positional encoding for transformer models."""

    def __init__(self, d_model: int, max_len: int = 32768):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(1) <= self.pe.size(1), f"Sequence too long ({x.size(1)} > {self.pe.size(1)})"

        # x: [B, L, D]
        pe = self.pe[:, : x.size(1), :].to(x.device)
        return x + pe


class SequenceCNNTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        d_model: int = 256,
        cnn_kernel_size: int = 3,
        dropout: float = 0.1,
        num_encoder_layers: int = 2,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        number_of_classes: int = 2,
        padding_idx: int = 4,
    ):
        super().__init__()

        self.number_of_classes = number_of_classes
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len)

        # CNN stack with 3 pooling layers for 8x reduction
        self.cnn = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=cnn_kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 2x reduction
            nn.Conv1d(d_model, d_model, kernel_size=cnn_kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 4x reduction
            nn.Conv1d(d_model, d_model, kernel_size=cnn_kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 8x reduction
        )

        self.norm = nn.LayerNorm(d_model)

        # Transformer encoder for global context (works on reduced length)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Attention pooling
        self.attn_pool = nn.Linear(d_model, 1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, number_of_classes),  # Binary classification
        )

        self.apply(self._init_weights)

    def forward(self, input_ids: torch.Tensor, input_quals: torch.Tensor | None = None):
        x = self.embedding(input_ids)

        # CNN downsampling
        x = x.transpose(1, 2)  # [B, D, L]
        x = self.cnn(x)
        x = x.transpose(1, 2)  # [B, L', D]

        x = self.pos_encoder(x)
        x = self.norm(x)

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Attention pooling
        attn_weights = torch.softmax(self.attn_pool(x), dim=1)  # [B, L', 1]
        pooled = torch.sum(attn_weights * x, dim=1)  # [B, D]

        return self.classifier(pooled)

    def _init_weights(self, module=None):
        module = module or self
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)
                if m.padding_idx is not None:
                    with torch.no_grad():
                        m.weight[m.padding_idx].fill_(0)
