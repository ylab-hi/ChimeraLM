import math

import torch
from einops import rearrange
from torch import nn


class HyenaOperator(nn.Module):
    def __init__(
        self,
        d_model: int,
        l_max: int = 20000,
        order: int = 2,
        filter_order: int = 64,
        num_heads: int = 1,
        inner_factor: float = 1.0,
        num_blocks: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.l_max = l_max
        self.order = order
        self.filter_order = filter_order
        self.num_heads = num_heads
        self.inner_factor = inner_factor
        self.num_blocks = num_blocks

        # Project input
        self.in_proj = nn.Linear(d_model, int(d_model * inner_factor))

        # Frequency filters
        self.filter_init = nn.Parameter(torch.randn(num_heads, filter_order) / math.sqrt(filter_order))

        # Length-dependent position embeddings
        self.pos_emb = nn.Parameter(torch.randn(1, l_max, d_model) / math.sqrt(d_model))

        # Output projection
        self.out_proj = nn.Linear(int(d_model * inner_factor), d_model)
        self.dropout = nn.Dropout(dropout)

        # Initialize filters
        self._init_filters()

    def _init_filters(self):
        # Initialize frequency domain filters
        filters = []
        for i in range(self.order):
            scale = 1.0 / (2**i)
            freq_response = torch.exp(-torch.arange(self.filter_order, dtype=torch.float32) * scale)
            filters.append(freq_response)
        self.register_buffer("filters", torch.stack(filters))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        H = self.num_heads

        # Project input
        x = self.in_proj(x)
        x = rearrange(x, "b l (h d) -> b h l d", h=H)

        # Apply position embeddings
        pos = self.pos_emb[:, :L]
        x = x + rearrange(pos, "1 l d -> 1 1 l d")

        # Frequency domain transformation
        x_f = torch.fft.rfft(x, dim=2)

        # Apply filters
        out = []
        for i in range(self.order):
            filter_response = self.filters[i].unsqueeze(0).unsqueeze(-1)
            y_f = x_f * filter_response
            y = torch.fft.irfft(y_f, n=L, dim=2)
            out.append(y)

        # Combine filtered outputs
        x = torch.sum(torch.stack(out), dim=0)

        # Reshape and project output
        x = rearrange(x, "b h l d -> b l (h d)")
        return self.dropout(self.out_proj(x))


class HyenaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        l_max: int = 20000,
        order: int = 2,
        filter_order: int = 64,
        num_heads: int = 4,
        inner_factor: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.hyena = HyenaOperator(
            d_model=d_model,
            l_max=l_max,
            order=order,
            filter_order=filter_order,
            num_heads=num_heads,
            inner_factor=inner_factor,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(4 * d_model)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(4 * d_model), d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Hyena operator with residual
        x = x + self.hyena(self.norm1(x))
        # MLP with residual
        return x + self.mlp(self.norm2(x))


class HyenaDNAClassifier(nn.Module):
    def __init__(
        self,
        seq_len: int,
        d_model: int = 256,
        n_layers: int = 4,
        num_heads: int = 4,
        order: int = 2,
        filter_order: int = 64,
        inner_factor: float = 2.0,
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.number_of_classes = num_classes

        # Embedding
        self.embedding = nn.Embedding(self.vocab_size, d_model)

        # Input normalization
        self.norm = nn.LayerNorm(d_model)

        # Hyena layers
        self.layers = nn.ModuleList(
            [
                HyenaBlock(
                    d_model=d_model,
                    l_max=seq_len,
                    order=order,
                    filter_order=filter_order,
                    num_heads=num_heads,
                    inner_factor=inner_factor,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        # Classification head
        self.classifier = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, num_classes))

    def forward(self, input_ids: torch.Tensor, input_quals: torch.Tensor | None = None) -> torch.Tensor:
        # Embed sequences
        x = self.embedding(input_ids)
        x = self.norm(x)

        # Process through Hyena layers
        for layer in self.layers:
            x = layer(x)

        # Global pooling and classification
        x = x.mean(dim=1)
        return self.classifier(x)
