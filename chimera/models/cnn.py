import torch
import torch.nn.functional as F
from torch import nn


class DNAConvNet(nn.Module):
    def __init__(
        self,
        seq_len: int,
        num_classes: int = 2,
        num_filters: list[int] = [256, 256, 256],
        kernel_sizes: list[int] = [7, 7, 7],
        pool_sizes: list[int] = [4, 4, 4],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.number_of_classes = num_classes

        # Input layer: one-hot encoding (4 channels for A,T,C,G)
        in_channels = 4

        # Create convolutional blocks
        self.conv_blocks = nn.ModuleList()
        current_len = seq_len

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

    def _one_hot_encode(self, sequences: torch.Tensor) -> torch.Tensor:
        """Convert integer sequences to one-hot encoding."""
        # sequences shape: [batch_size, seq_len]
        # output shape: [batch_size, 4, seq_len]
        return F.one_hot(sequences, num_classes=4).float().transpose(1, 2)

    def forward(self, input_ids: torch.Tensor, input_quals: torch.Tensor | None = None) -> torch.Tensor:
        # One-hot encode the sequences
        x = self._one_hot_encode(input_ids)

        # Apply convolution blocks
        for block in self.conv_blocks:
            x = block(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Apply fully connected layers
        return self.fc(x)
