import math

import torch
from torch import nn

# https://github.com/state-spaces/mamba


class SelectiveScan(nn.Module):
    def __init__(self, d_model, d_state=16, dt_rank=None, dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank if dt_rank is not None else math.ceil(d_model / 8)

        # A, B, C matrices for selective scan
        self.A = nn.Parameter(torch.randn(self.d_model, self.d_state) / math.sqrt(d_state))
        self.B = nn.Parameter(torch.randn(self.d_model, self.d_state) / math.sqrt(d_state))
        self.C = nn.Parameter(torch.randn(self.d_model, self.d_state) / math.sqrt(d_model))

        # ∆ projection
        self.dt_projs = nn.Parameter(torch.randn(self.dt_rank, d_model) / math.sqrt(d_model))
        self.dt_init = math.log(dt_max / dt_min) / self.dt_rank

    def forward(self, x: torch.Tensor, delta: torch.Tensor | None = None) -> torch.Tensor:
        batch, length, dim = x.shape

        # Compute ∆
        if delta is None:
            delta = torch.exp(torch.arange(self.dt_rank, device=x.device) * (-self.dt_init))
            delta = torch.einsum("r,rd->d", delta, self.dt_projs)
            delta = delta.exp()

        # Compute scan
        x_dbl = torch.cat([torch.zeros_like(x[:, :1]), x], dim=1)
        u = torch.einsum("bld,de->ble", x_dbl, self.A)
        v = torch.einsum("bld,de->ble", x, self.B)

        # Parallel scan
        uu = u[:, 1:]
        vv = v
        last_state = None
        for i in range(length):
            last_state = uu[:, i : i + 1] if last_state is None else last_state * delta + uu[:, i : i + 1]
            vv[:, i : i + 1] = vv[:, i : i + 1] + torch.einsum("ble,de->bld", last_state, self.C)
        return vv


class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
        dt_rank: int | None = None,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand_factor = expand_factor
        self.d_inner = int(expand_factor * d_model)

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)

        # Convolution
        self.conv = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )

        # Selective scan
        self.scan = SelectiveScan(d_model=self.d_inner, d_state=d_state, dt_rank=dt_rank, dt_min=dt_min, dt_max=dt_max)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model)

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input projection and splitting
        residual = x
        x = self.norm(x)
        x, gate = self.in_proj(x).chunk(2, dim=-1)

        # Convolution
        x = x.transpose(1, 2)
        x = self.conv(x)[..., : -self.conv.kernel_size[0] + 1]
        x = x.transpose(1, 2)

        # Selective scan
        x = self.scan(x)
        # Gating and output
        x = x * torch.nn.functional.silu(gate)
        x = self.out_proj(x)

        return x + residual


class MambaDNAClassifier(nn.Module):
    def __init__(
        self,
        seq_len: int,
        d_model: int = 256,
        n_layers: int = 4,
        d_state: int = 16,
        expand_factor: int = 2,
        num_classes: int = 2,
    ):
        super().__init__()
        self.number_of_classes = num_classes

        # Embedding
        self.embedding = nn.Embedding(self.vocab_size, d_model)

        # Mamba layers
        self.layers = nn.ModuleList(
            [MambaBlock(d_model=d_model, d_state=d_state, expand_factor=expand_factor) for _ in range(n_layers)]
        )

        # Classification head
        self.classifier = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_classes))

    def forward(self, input_ids: torch.Tensor, input_quals: torch.Tensor | None = None) -> torch.Tensor:
        # Embed sequences
        x = self.embedding(input_ids)

        # Process through Mamba layers
        for layer in self.layers:
            x = layer(x)

        # Global pooling and classification
        x = x.mean(dim=1)
        return self.classifier(x)
