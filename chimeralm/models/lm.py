from functools import partial

import torch

from .basic_module import ClassificationLit
from .components import hyena


class ChimeraLM:
    """ChimeraLM model for sequence classification tasks."""

    @classmethod
    def from_pretrained(
        cls, model_name: str = "yangliz5/chimeralm", *, save_attention: bool = False
    ) -> ClassificationLit:
        """Load a pretrained ChimeraLM model from the Hugging Face model hub."""
        return ClassificationLit.from_pretrained(
            model_name,
            net=hyena.HyenaDna(
                number_of_classes=2,
                backbone_name="hyenadna-small-32k-seqlen",
                head=hyena.BinarySequenceClassifier(
                    input_dim=256,
                    hidden_dim=512,
                    num_layers=2,
                    dropout=0.1,
                    pooling_type="attention",
                    activation="gelu",
                    use_residual=True,
                    save_attention=save_attention,
                ),
            ),
            optimizer=partial(torch.optim.AdamW, lr=0.0001, weight_decay=0.01),
            scheduler=partial(torch.optim.lr_scheduler.ReduceLROnPlateau, mode="min", factor=0.1, patience=10),
            criterion=torch.nn.CrossEntropyLoss(),
            compile=False,
        )

    @classmethod
    def new(cls, *, save_attention: bool = False) -> ClassificationLit:
        """Load a new ChimeraLM model."""
        return ClassificationLit(
            net=hyena.HyenaDna(
                number_of_classes=2,
                backbone_name="hyenadna-small-32k-seqlen",
                head=hyena.BinarySequenceClassifier(
                    input_dim=256,
                    hidden_dim=512,
                    num_layers=2,
                    dropout=0.1,
                    pooling_type="attention",
                    activation="gelu",
                    use_residual=True,
                    save_attention=save_attention,
                ),
            ),
            optimizer=partial(torch.optim.AdamW, lr=0.0001, weight_decay=0.01),
            scheduler=partial(torch.optim.lr_scheduler.ReduceLROnPlateau, mode="min", factor=0.1, patience=10),
            criterion=torch.nn.CrossEntropyLoss(),
            compile=False,
        )
