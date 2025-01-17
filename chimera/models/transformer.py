import math

import torch
from torch import nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer


class SequenceTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        """Transformer model for sequence classification with encoder-decoder architecture.

        Args:
            vocab_size: Size of the vocabulary
            d_model: Dimension of the model
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
        """
        super().__init__()

        self.number_of_classes = 2  # Binary classification
        self.d_model = d_model

        # Special tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Embedding layers for sequence and quality scores
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.qual_linear = nn.Linear(1, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer encoder
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)

        # Transformer decoder
        decoder_layers = TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer_decoder = TransformerDecoder(decoder_layers, num_decoder_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, self.number_of_classes),
        )

    def forward(self, input_ids: torch.Tensor, input_quals: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            input_ids: Tensor of shape [batch_size, seq_length]
            input_quals: Tensor of shape [batch_size, seq_length]

        Returns:
            Tensor of shape [batch_size, num_classes]
        """
        batch_size = input_ids.size(0)

        # Create embeddings
        token_embeds = self.token_embedding(input_ids)
        qual_embeds = self.qual_linear(input_quals.unsqueeze(-1))

        # Combine embeddings
        x = token_embeds + qual_embeds

        # Prepend CLS token to the sequence
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Create padding mask (accounting for CLS token)
        padding_mask = torch.cat(
            [torch.zeros(batch_size, 1, device=input_ids.device, dtype=torch.bool), (input_ids == 0)], dim=1
        )

        # Pass through encoder
        memory = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        # Create decoder input (using CLS token)
        decoder_input = cls_tokens

        # Pass through decoder
        decoder_output = self.transformer_decoder(decoder_input, memory, memory_key_padding_mask=padding_mask)

        # Use the decoder output for classification
        logits = self.classifier(decoder_output.squeeze(1))

        return logits


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """Positional encoding for transformer model.

        Args:
            d_model: Dimension of the model
            dropout: Dropout probability
            max_len: Maximum sequence length
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Model forward pass.

        Args:
            x: Tensor of shape [batch_size, seq_length, embedding_dim]

        Returns:
            Tensor of shape [batch_size, seq_length, embedding_dim]
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)
