import logging
from dataclasses import dataclass
from pathlib import Path

import lightning
import torch
import typer
from click import Context
from rich import print
from rich.logging import RichHandler
from typer.core import TyperGroup

import chimera


@dataclass
class ModelConfig:
    """Configuration for the Mamba model."""

    embedding_dim: int
    number_of_layers: int
    number_of_classes: int
    dropout: float
    d_state: int = 128
    d_conv: int = 4
    expand: int = 2
    headdim: int = 64
    vocab_size: int = 12


def create_model(config: ModelConfig) -> torch.nn.Module:
    """Create a Mamba model using the given configuration.

    Parameters:
        config (ModelConfig): The configuration to use for the model.

    Returns:
        torch.nn.Module: The created model.
    """
    return chimera.models.components.mamba.MambaSequenceClassificationSP(
        vocab_size=config.vocab_size,
        embedding_dim=config.embedding_dim,
        number_of_layers=config.number_of_layers,
        dropout=config.dropout,
        number_of_classes=config.number_of_classes,
        d_state=config.d_state,
        d_conv=config.d_conv,
        expand=config.expand,
        headdim=config.headdim,
    )


def set_logging_level(level: int = logging.INFO):
    """Set the logging level.

    Parameters:
        level (int): The logging level to set.
    """
    FORMAT = "%(message)s"
    logging.basicConfig(
        level=level,
        format=FORMAT,
        handlers=[RichHandler()],
    )


def predict(
    data_path: Path = typer.Argument(..., help="Path to the dataset"),
    gpus: int = typer.Option(0, "--gpus", "-g", help="Number of GPUs to use"),
    output_path: Path | None = typer.Option(None, "--output", "-o", help="Output path for predictions"),
    batch_size: int = typer.Option(12, "--batch-size", "-b", help="Batch size"),
    num_workers: int = typer.Option(0, "--workers", "-w", help="Number of workers"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    max_sample: int | None = typer.Option(None, "--max-sample", "-m", help="Maximum number of samples to process"),
    limit_predict_batches: int | None = typer.Option(None, "--limit-batches", "-l", help="Limit prediction batches"),
):
    """Predict the given dataset using DeepChopper."""
    if verbose:
        set_logging_level(logging.INFO)

    if isinstance(data_path, str):
        data_path = Path(data_path)

    tokenizer = chimera.data.tokenizer.CharacterTokenizer()
    datamodule: lightning.LightningDataModule = chimera.data.fq.DataModule(
        train_data_path="dummy.parquet",
        tokenizer=tokenizer,
        predict_data_path=data_path.as_posix(),
        batch_size=batch_size,
        num_workers=num_workers,
        max_predict_samples=max_sample,
    )

    model = create_model(ModelConfig(embedding_dim=512, number_of_layers=6, number_of_classes=2, dropout=0.1))

    output_path = Path(output_path or "predictions")
    callbacks = [chimera.models.callbacks.CustomWriter(output_dir=output_path, write_interval="batch")]

    if gpus > 0:
        if torch.cuda.is_available():
            accelerator = "gpu"
            devices = min(gpus, torch.cuda.device_count())
        elif torch.backends.mps.is_available():
            accelerator = "mps"
            devices = "auto"  # MPS currently supports only one device
        else:
            accelerator = "cpu"
            devices = "auto"
    else:
        accelerator = "cpu"
        devices = "auto"

    trainer = lightning.pytorch.trainer.Trainer(
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        deterministic=False,
        logger=False,
        limit_predict_batches=limit_predict_batches,
    )

    import multiprocess.context as ctx

    ctx._force_start_method("spawn")
    trainer.predict(model=model, dataloaders=datamodule, return_predictions=False)


class OrderCommands(TyperGroup):
    """Order commands in the order appear."""

    def list_commands(self, ctx: Context):
        """Return list of commands in the order appear."""
        return list(self.commands)  # get commands using self.commands


def version_callback(value: bool):
    """Print the version and exit."""
    if value:
        print(f"DeepChopper Version: {chimera.__version__}")
        raise typer.Exit()


app = typer.Typer(
    cls=OrderCommands,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="DeepChopper: A genomic lanuage model to identify artificial sequences.",
)


# Add the version option to the main app
@app.callback()
def main(
    version: bool | None = typer.Option(
        None,
        "--version",
        "-V",
        help="Show the application's version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
):
    """Main entry point for the Chimera CLI."""


if __name__ == "__main__":
    main()
