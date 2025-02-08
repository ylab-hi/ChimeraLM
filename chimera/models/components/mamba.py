import torch
from mamba_ssm import Mamba2
from torch import Tensor, nn

# https://github.com/state-spaces/mamba


class MambaSequenceClassification(nn.Module):
    """Mamba model for sequence classification tasks with multiple Mamba layers."""

    def __init__(
        self,
        vocab_size,
        embedding_dim: int,
        number_of_layers: int,
        model_max_length: int,
        dropout: float,
        number_of_classes: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        padding_idx: int = 4,
    ):
        super().__init__()
        self.number_of_classes = number_of_classes

        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

        self.pos_embedding = nn.Parameter(torch.zeros(1, model_max_length, embedding_dim))

        # Input projection, normalization and dropout combined
        self.input_block = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim), nn.LayerNorm(embedding_dim), nn.Dropout(dropout)
        )

        # Stack of Mamba layers with skip connections
        self.mamba_layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "mamba": Mamba2(
                            d_model=embedding_dim, d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim
                        ),
                        "dropout": nn.Dropout(dropout),
                    }
                )
                for _ in range(number_of_layers)
            ]
        )

        # Output head
        self.pooler = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.GELU(), nn.Dropout(dropout))
        # Classification head with multiple layers
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, number_of_classes),
        )

        # Initialize embeddings
        nn.init.normal_(self.pos_embedding, std=0.02)

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, seq_len)
            attention_mask: Optional mask tensor of shape (batch_size, seq_len)

        Returns:
            Classification logits of shape (batch_size, n_classes)
        """
        # Embed input
        x = self.embedding(x)  # (batch, seq_len, d_model)
        x = x + self.pos_embedding[:, : x.size(1), :]

        # Initial projection and normalization
        x = self.input_block(x)

        # Apply mask if provided
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)

        # Pass through Mamba layers with skip connections
        for layer in self.mamba_layers:
            residual = x
            # Mamba block
            x = layer["mamba"](x)
            x = layer["dropout"](x)
            x = residual + x

            # Apply mask after each layer if provided
            if attention_mask is not None:
                x = x * attention_mask.unsqueeze(-1)

        # Advanced pooling - combine max and mean pooling
        mean_pooled = x.mean(dim=1)  # (batch, d_model)
        max_pooled = x.max(dim=1)[0]  # (batch, d_model)
        pooled = (mean_pooled + max_pooled) / 2

        # Final classification
        pooled = self.pooler(pooled)
        return self.classifier(pooled)


class MambaSequenceClassificationSP(nn.Module):
    """Mamba model for sequence classification tasks with multiple Mamba layers."""

    def __init__(
        self,
        vocab_size,
        embedding_dim: int,
        number_of_layers: int,
        number_of_classes: int,
        dropout: float,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        padding_idx: int = 4,
    ):
        super().__init__()
        self.number_of_classes = number_of_classes
        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

        # Stack of Mamba layers with skip connections
        self.mamba_layers = nn.ModuleList(
            [
                Mamba2(
                    d_model=embedding_dim,  # Input/output dimension
                    d_state=d_state,  # Internal state dimension
                    d_conv=d_conv,  # Convolution width
                    expand=expand,  # Expansion factor - controls width of the block's internal feed-forward network
                    headdim=headdim,
                )
                for _ in range(number_of_layers)
            ]
        )

        # Output head
        self.pooler = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.GELU(), nn.Dropout(dropout))

        # Classification head with multiple layers
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, number_of_classes),
        )

    def forward(
        self,
        input_ids: Tensor,
        input_quals: Tensor | None = None,
    ) -> Tensor:
        """Forward pass of the model.

        Args:
            input_ids: Input tensor of shape (batch_size, seq_len)
            input_quals: Optional quality scores tensor of shape (batch_size, seq_len)

        Returns:
            Classification logits of shape (batch_size, n_classes)
        """
        # Embed input
        x = self.embedding(input_ids)  # (batch, seq_len, d_model)
        # Pass through Mamba layers with skip connections
        for layer in self.mamba_layers:
            residual = x
            # Mamba block
            x = layer(x)
            x = residual + x

        # Advanced pooling - combine max and mean pooling
        mean_pooled = x.mean(dim=1)  # (batch, d_model)
        max_pooled = x.max(dim=1)[0]  # (batch, d_model)
        pooled = (mean_pooled + max_pooled) / 2

        # Final classification
        pooled = self.pooler(pooled)
        return self.classifier(pooled)
