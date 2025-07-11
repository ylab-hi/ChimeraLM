import torch
import torch.nn.functional as F
from torch import nn

class SequenceCNNTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        d_model: int = 256,
        cnn_kernel_size: int = 3,
        dropout: float = 0.1,
        num_encoder_layers: int = 2,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        padding_idx: int = 4,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.qual_linear = nn.Linear(1, d_model)

        # Learnable positional embedding
        self.pos_embedding = nn.Embedding(max_len, d_model)

        # CNN stack with 3 pooling layers for 8x reduction
        self.cnn = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=cnn_kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 2x reduction

            nn.Conv1d(d_model, d_model, kernel_size=cnn_kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 4x reduction

            nn.Conv1d(d_model, d_model, kernel_size=cnn_kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 8x reduction
        )

        self.norm = nn.LayerNorm(d_model)

        # Optional transformer encoder for global context (works on reduced length)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Attention pooling
        self.attn_pool = nn.Linear(d_model, 1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),  # Binary classification
        )

        self.apply(self._init_weights)

    def forward(self, input_ids: torch.Tensor, input_quals: torch.Tensor | None = None):
        x = self.embedding(input_ids)

        if input_quals is not None:
            qual_embeds = self.qual_linear(input_quals.unsqueeze(-1))
            x += qual_embeds

        # Positional embedding
        seq_len = x.size(1)
        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x += self.pos_embedding(pos_ids)

        # Handle input padding for divisibility by 8
        pad_len = (8 - (seq_len % 8)) % 8
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))  # pad sequence dim

        # CNN downsampling
        x = x.transpose(1, 2)  # [B, D, L]
        x = self.cnn(x)
        x = x.transpose(1, 2)  # [B, L', D]

        x = self.norm(x)

        # Transformer encoder (optional)
        x = self.transformer_encoder(x)

        # Attention pooling
        attn_weights = torch.softmax(self.attn_pool(x), dim=1)  # [B, L', 1]
        pooled = torch.sum(attn_weights * x, dim=1)  # [B, D]

        return self.classifier(pooled)

    def _init_weights(self, module=None):
        module = module or self
        for p in module.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)