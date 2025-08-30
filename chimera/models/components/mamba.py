import contextlib

import torch
from torch import Tensor, nn

# https://github.com/state-spaces/mamba
with contextlib.suppress(ImportError):
    from mamba_ssm import Mamba2


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


class MambaSequenceClassificationPositional(nn.Module):
    """Mamba model for sequence classification tasks with multiple Mamba layers."""

    def __init__(
        self,
        vocab_size,
        embedding_dim: int,
        number_of_layers: int,
        number_of_classes: int,
        dropout: float,
        max_seq_length: int,  # Add max sequence length parameter
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        padding_idx: int = 4,
        pos_embedding_type: str = "learned",  # "learned", "sinusoidal", or "none"
    ):
        super().__init__()
        self.number_of_classes = number_of_classes
        self.max_seq_length = max_seq_length
        self.pos_embedding_type = pos_embedding_type

        # Token embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

        # Positional embedding layer
        if pos_embedding_type == "learned":
            self.position_embedding = nn.Embedding(max_seq_length, embedding_dim)
        elif pos_embedding_type == "sinusoidal":
            self.register_buffer(
                "position_embedding", self._create_sinusoidal_embeddings(max_seq_length, embedding_dim)
            )

        # Dropout for embeddings
        self.embedding_dropout = nn.Dropout(dropout)

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

        # Initialize embeddings
        self._init_embeddings()

    def _create_sinusoidal_embeddings(self, max_seq_length: int, embedding_dim: int) -> Tensor:
        """Create sinusoidal positional embeddings."""
        pe = torch.zeros(max_seq_length, embedding_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def _init_embeddings(self):
        """Initialize embedding weights."""
        # Initialize token embeddings
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        if self.embedding.padding_idx is not None:
            nn.init.constant_(self.embedding.weight[self.embedding.padding_idx], 0)

        # Initialize learned positional embeddings if using them
        if self.pos_embedding_type == "learned":
            nn.init.normal_(self.position_embedding.weight, mean=0, std=0.02)

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
            Classification logits of shape (batch_size, n_classes).
        """
        batch_size, seq_len = input_ids.shape

        # Check sequence length
        if seq_len > self.max_seq_length:
            msg = f"Sequence length {seq_len} exceeds maximum length {self.max_seq_length}"
            raise ValueError(msg)

        # Token embeddings
        x = self.embedding(input_ids)  # (batch, seq_len, d_model)

        # Add positional embeddings
        if self.pos_embedding_type == "learned":
            # Create position indices
            positions = torch.arange(seq_len, device=input_ids.device)
            pos_embeddings = self.position_embedding(positions)  # (seq_len, d_model)
            x = x + pos_embeddings.unsqueeze(0)  # Broadcast to (batch, seq_len, d_model)

        elif self.pos_embedding_type == "sinusoidal":
            pos_embeddings = self.position_embedding[:seq_len]  # (seq_len, d_model)
            x = x + pos_embeddings.unsqueeze(0)  # Broadcast to (batch, seq_len, d_model)

        # Apply embedding dropout
        x = self.embedding_dropout(x)

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
