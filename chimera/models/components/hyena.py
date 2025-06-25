import torch
from torch import nn
from transformers import AutoModel


class Classifier(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        number_of_classes: int,
        input_dim: int,  # 256 from Hyena
        hidden_dim1: int,
        hidden_dim2: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.number_of_classes = number_of_classes
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2, number_of_classes),
        )

    def forward(self, hidden_states: torch.Tensor, input_quals: torch.Tensor | None = None):
        return self.net(hidden_states)


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
        return self.head(hidden_states, input_quals)
