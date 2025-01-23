import torch
from torch import nn


class DNAConvNet(nn.Module):
    def __init__(
        self,
        vocab_size: int,  # Size of vocabulary including special tokens
        number_of_classes: int = 2,
        embedding_dim: int = 256,  # Embedding dimension
        num_filters: list[int] | None = None,
        kernel_sizes: list[int] | None = None,
        pool_sizes: list[int] | None = None,
        dropout: float = 0.1,
        padding_idx: int = 4,  # Padding token ID
    ):
        if pool_sizes is None:
            pool_sizes = [4, 4, 4]
        if kernel_sizes is None:
            kernel_sizes = [7, 7, 7]
        if num_filters is None:
            num_filters = [256, 256, 256]
        super().__init__()
        self.number_of_classes = number_of_classes

        # Embedding layer instead of one-hot encoding
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=padding_idx)

        # Create convolutional blocks
        self.conv_blocks = nn.ModuleList()
        in_channels = embedding_dim

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

        # Adaptive pooling to handle variable sequence lengths
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(num_filters[-1], 256),  # Input size is now just the number of filters
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, number_of_classes),
        )

    def forward(self, input_ids: torch.Tensor, input_quals: torch.Tensor | None = None) -> torch.Tensor:
        # Embed the sequence [batch_size, seq_len] -> [batch_size, seq_len, embed_dim]
        x = self.embedding(input_ids)

        # Transpose for convolution [batch_size, seq_len, embed_dim] -> [batch_size, embed_dim, seq_len]
        x = x.transpose(1, 2)

        # Apply convolution blocks
        for block in self.conv_blocks:
            x = block(x)

        # Global pooling
        x = self.adaptive_pool(x)
        x = x.squeeze(-1)

        # Apply fully connected layers
        return self.fc(x)
