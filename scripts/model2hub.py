import chimera 
from chimera.models import hyena

from functools import partial
import torch

import typer
import ipdb

app = typer.Typer()

@app.command()
def main(ckpt_path: str, model_name: str="yangliz5/chimeralm"):
    model = chimera.models.basic_module.ClassificationLit(net=hyena.HyenaDna(
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
                ),
            ),
            optimizer=partial(torch.optim.AdamW, lr=0.0001, weight_decay=0.01),
            scheduler=partial(torch.optim.lr_scheduler.ReduceLROnPlateau, mode="min", factor=0.1, patience=10),
            criterion=torch.nn.CrossEntropyLoss(),
            compile=False,
        )

    model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["state_dict"])
    model.push_to_hub(model_name)

if __name__ == "__main__":
    app()

