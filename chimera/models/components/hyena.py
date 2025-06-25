import torch
import torch.nn as nn
from transformers import AutoModel


class LinearHead(nn.Module):
    def __init__(self, number_of_classes: int, hidden_dim1: int, hidden_dim2: int, hidden_dim3: int,
    ):
        super().__init__()
        self.number_of_classes = number_of_classes
        self.net = nn.Sequential(
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.GELU(),
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.GELU(),
            nn.Linear(hidden_dim3, number_of_classes),
        )

    def forward(self, hidden_states: torch.Tensor, input_quals: torch.Tensor):
        return self.net(hidden_states)


class HeynaDna(nn.Module):
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
        input_quals: torch.Tensor,
    ):
        transformer_outputs = self.backbone(
            input_ids,
            inputs_embeds=None,
            output_hidden_states=None,
            return_dict=None,
        )
        hidden_states = transformer_outputs[0]
        return self.head(hidden_states, input_quals)
