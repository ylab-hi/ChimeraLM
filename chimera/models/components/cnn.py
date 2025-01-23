import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class DNAConvNet(nn.Module):
    def __init__(
        self,
        vocab_size: int,  # Size of vocabulary including special tokens
        num_classes: int = 2,
        seq_len: int = 20000,
        embed_dim: int = 256,  # Embedding dimension
        num_filters: list[int] = [256, 256, 256],
        kernel_sizes: list[int] = [7, 7, 7],
        pool_sizes: list[int] = [4, 4, 4],
        dropout: float = 0.1,
        padding_idx: int = 4,  # Padding token ID
    ):
        super().__init__()
        self.number_of_classes = num_classes

        # Embedding layer instead of one-hot encoding
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=padding_idx)

        # Create convolutional blocks
        self.conv_blocks = nn.ModuleList()
        current_len = seq_len
        in_channels = embed_dim

        for filters, kernel_size, pool_size in zip(num_filters, kernel_sizes, pool_sizes, strict=False):
            block = nn.Sequential(
                nn.Conv1d(in_channels, filters, kernel_size, padding="same"),
                nn.BatchNorm1d(filters),
                nn.GELU(),
                nn.MaxPool1d(pool_size),
                nn.Dropout(dropout),
            )
            self.conv_blocks.append(block)
            in_channels = filters
            current_len = current_len // pool_size

        # Calculate the flattened size
        self.flat_size = current_len * num_filters[-1]

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.flat_size, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids: torch.Tensor, input_quals: torch.Tensor | None = None) -> torch.Tensor:
        # Embed the sequence [batch_size, seq_len] -> [batch_size, seq_len, embed_dim]
        x = self.embedding(input_ids)

        # Transpose for convolution [batch_size, seq_len, embed_dim] -> [batch_size, embed_dim, seq_len]
        x = x.transpose(1, 2)

        # Apply convolution blocks
        for block in self.conv_blocks:
            x = block(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Apply fully connected layers
        return self.fc(x)
