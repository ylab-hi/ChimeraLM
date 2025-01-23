import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 20000):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
        return self.cos_cached, self.sin_cached


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of a tensor by 90 degrees."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class LinearAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        inner_dim = heads * dim_head

        self.to_qkv = nn.Linear(dim, inner_dim * 3)
        self.to_out = nn.Linear(inner_dim, dim)
        self.rotary_pos_emb = RotaryPositionalEmbedding(dim_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        h = self.heads

        # Project to queries, keys, values
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (t.reshape(b, n, h, self.dim_head) for t in qkv)

        # Apply rotary position embeddings
        cos, sin = self.rotary_pos_emb(q, n)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin

        # Linear attention
        q = F.elu(q) + 1
        k = F.elu(k) + 1

        # Compute attention
        q = q.transpose(1, 2)  # [b, h, n, d]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        kv = torch.matmul(k.transpose(-2, -1), v)  # [b, h, d, d]
        qkv = torch.matmul(q, kv)  # [b, h, n, d]

        # Normalize
        normalizer = torch.matmul(q, k.transpose(-2, -1).sum(dim=-2, keepdim=True))
        out = qkv / (normalizer + 1e-8)

        # Merge heads and project
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


class LocalAttentionBlock(nn.Module):
    def __init__(self, dim: int, window_size: int = 512):
        super().__init__()
        self.window_size = window_size
        self.attention = LinearAttention(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split into windows
        b, n, d = x.shape
        pad_size = (self.window_size - n % self.window_size) % self.window_size
        padded = F.pad(x, (0, 0, 0, pad_size))
        windows = padded.view(b, -1, self.window_size, d)

        # Apply attention to each window
        windows = windows + self.attention(self.ln1(windows))
        windows = windows + self.mlp(self.ln2(windows))

        # Merge windows back
        out = windows.view(b, -1, d)
        return out[:, :n, :]


class DNAClassifierNet(nn.Module):
    """Neural network for DNA sequence classification."""

    def __init__(
        self,
        seq_len: int,
        vocab_size: int,
        embed_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        window_size: int = 512,
        num_classes: int = 2,
        padding_idx: int = 4,
    ):
        super().__init__()
        self.number_of_classes = num_classes
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=padding_idx)

        # Hierarchical processing with local attention
        self.local_layers = nn.ModuleList(
            [LocalAttentionBlock(dim=embed_dim, window_size=window_size) for _ in range(num_layers)]
        )

        # Global processing with linear attention
        self.global_attention = LinearAttention(dim=embed_dim, heads=num_heads)

        # Classification head
        self.classifier = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))

    def forward(self, input_ids: torch.Tensor, input_quals: torch.Tensor | None = None) -> torch.Tensor:
        # Embed sequences
        x = self.embedding(input_ids)

        # Local processing
        for layer in self.local_layers:
            x = layer(x)

        # Global processing
        x = self.global_attention(x)

        # Global pooling and classification
        x = x.mean(dim=1)
        return self.classifier(x)
