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
        max_length: int,
        dropout: float,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        number_of_classes: int = 2,
        padding_idx: int = 4,
    ):
        super().__init__()
        self.number_of_classes = number_of_classes

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_length, embedding_dim))

        # Stack of Mamba layers
        self.layers = nn.ModuleList(
            [
                Mamba2(
                    d_model=embedding_dim,
                    d_state=d_state,  # Default state dimension
                    d_conv=d_conv,  # Default conv dimension
                    expand=expand,  # Default expansion factor
                )
                for _ in range(number_of_layers)
            ]
        )

        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, number_of_classes)

        # Initialize position embeddings
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

        # Apply mask if provided
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)

        # Pass through Mamba layers
        for layer in self.layers:
            # Residual connection
            residual = x
            x = layer(x)
            x = x + residual
            x = self.layer_norm(x)

            if attention_mask is not None:
                x = x * attention_mask.unsqueeze(-1)

        # Pool sequence - use mean pooling
        x = x.mean(dim=1)  # (batch, d_model)

        # Classify
        x = self.dropout(x)
        return self.classifier(x)  # (batch, n_classes)
