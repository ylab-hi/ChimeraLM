import torch
from torch import nn
from transformers import AutoModel


class BinarySequenceClassifier(nn.Module):
    """A binary classifier head specifically designed for sequence classification tasks.

    This classifier includes:
    - Sequence pooling (mean, max, or attention-based)
    - Multiple hidden layers with residual connections
    - Dropout for regularization
    - Binary output with optional sigmoid activation
    """

    def __init__(
        self,
        input_dim: int,  # Hidden dimension from backbone (e.g., 256 from Hyena)
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        pooling_type: str = "attention",  # "mean", "max", "attention", "cls"
        activation: str = "gelu",
        *,
        use_residual: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.pooling_type = pooling_type
        self.use_residual = use_residual

        # Activation function
        if activation == "gelu":
            self.activation: nn.GELU | nn.ReLU | nn.SiLU = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "swish":
            self.activation = nn.SiLU()
        else:
            msg = f"Unsupported activation: {activation}"
            raise ValueError(msg)

        # Attention-based pooling
        if pooling_type == "attention":
            self.attention = nn.Sequential(
                nn.Linear(input_dim, hidden_dim // 2), self.activation, nn.Linear(hidden_dim // 2, 1), nn.Softmax(dim=1)
            )

        # Classification layers
        layers = []
        prev_dim = input_dim

        for i in range(num_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout))

            if use_residual and i > 0 and prev_dim == hidden_dim:
                # Add residual connection
                layers.append(ResidualBlock(hidden_dim, dropout))
            else:
                prev_dim = hidden_dim

        self.classifier = nn.Sequential(*layers)

        # Final output layer for binary classification
        self.output_layer = nn.Linear(hidden_dim, 2)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None):
        _batch_size, _seq_len, _hidden_dim = hidden_states.shape

        # Apply pooling to get sequence representation
        if self.pooling_type == "mean":
            if attention_mask is not None:
                # Masked mean pooling
                mask_expanded = attention_mask.unsqueeze(-1).float()
                masked_hidden = hidden_states * mask_expanded
                sum_hidden = masked_hidden.sum(dim=1)
                seq_lengths = attention_mask.sum(dim=1, keepdim=True).float()
                pooled_output = sum_hidden / seq_lengths
            else:
                pooled_output = hidden_states.mean(dim=1)

        elif self.pooling_type == "max":
            if attention_mask is not None:
                # Masked max pooling
                mask_expanded = attention_mask.unsqueeze(-1).float()
                masked_hidden = hidden_states * mask_expanded
                # Set masked positions to large negative values
                masked_hidden = masked_hidden + (1 - mask_expanded) * -1e9
                pooled_output = masked_hidden.max(dim=1)[0]
            else:
                pooled_output = hidden_states.max(dim=1)[0]

        elif self.pooling_type == "attention":
            # Attention-based pooling
            attention_weights = self.attention(hidden_states)  # (batch_size, seq_len, 1)
            if attention_mask is not None:
                # Mask attention weights for padding tokens
                mask_expanded = attention_mask.unsqueeze(-1).float()
                attention_weights = attention_weights * mask_expanded
                # Renormalize attention weights
                attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-9)

            pooled_output = (hidden_states * attention_weights).sum(dim=1)

        elif self.pooling_type == "cls":
            # Use the first token (CLS token) representation
            pooled_output = hidden_states[:, 0, :]

        else:
            msg = f"Unsupported pooling type: {self.pooling_type}"
            raise ValueError(msg)

        # Pass through classifier layers
        features = self.classifier(pooled_output)

        # Final binary classification
        return self.output_layer(features)


class ResidualBlock(nn.Module):
    """Simple residual block for the classifier."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.layers(x)
        out = self.dropout(out)
        return out + residual


class QualLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()

        self.qual_linear = nn.Linear(1, hidden_dim)
        self.qual_linear_2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden_states: torch.Tensor, input_quals: torch.Tensor):
        qual_embeds = self.qual_linear(input_quals.unsqueeze(-1))
        return self.qual_linear_2(qual_embeds)


class HyenaDna(nn.Module):
    def __init__(
        self,
        number_of_classes: int,
        head: nn.Module,
        backbone_name: str = "hyenadna-small-32k-seqlen",
        *,
        freeze_backbone=False,
    ):
        super().__init__()
        self.number_of_classes = number_of_classes
        self.backbone_name = backbone_name
        self.backbone = AutoModel.from_pretrained(f"LongSafari/{backbone_name}-hf", trust_remote_code=True)
        self.head = head

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        input_quals: torch.Tensor | None = None,
    ):
        transformer_outputs = self.backbone(
            input_ids,
            inputs_embeds=None,
            output_hidden_states=None,
            return_dict=None,
        )
        hidden_states = transformer_outputs[0]
        return self.head(hidden_states, None)
