import logging
from collections import Counter
from pathlib import Path

import lightning
import multiprocess.context as ctx
import pysam
import torch
import typer
from click import Context
from rich.logging import RichHandler
from rich.progress import track
from typer.core import TyperGroup
import chimeralm


def load_predicts(path: Path | str) -> dict[str, int]:
    """Load predictions from a text file.

    Args:
        path: Path to the input file

    Returns:
        List of Predict objects
    """
    predicts = {}
    try:
        path = Path(path)
        if not path.exists():
            msg = f"File not found: {path}"
            raise FileNotFoundError(msg)

        with path.open(encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split("\t")
                if len(parts) != 2:
                    msg = f"Invalid line format at line {line_num}: {line}"
                    raise ValueError(msg)

                name, label_str = parts
                label = int(label_str)
                predicts[name] = label

    except Exception as e:
        msg = f"Error reading file {path}: {e}"
        raise ValueError(msg)

    return predicts


def load_predictions_from_folder(path: Path | str) -> dict[str, int]:
    """Load predictions from a folder."""
    predictions = {}
    for file in Path(path).glob("*.txt"):
        predictions.update(load_predicts(file))
    return predictions


def filter_bam_by_predcition(
    bam_path: Path, prediction_path: Path, *, progress_bar: bool = False, index: bool = True
) -> None:
    """Filter a BAM file by predictions.

    use rich progress bar if progress_bar is True
    """
    predictions = load_predictions_from_folder(prediction_path)
    logging.info(f"Loaded {len(predictions)} predictions from {prediction_path}")

    # summar 0 and 1 predictions
    counter = Counter(predictions.values())
    logging.info(
        f"Biological: {counter.get(0, 0)} ({counter.get(0, 0) / len(predictions) * 100:.1f}%), Chimera artifact: {counter.get(1, 0)} ({counter.get(1, 0) / len(predictions) * 100:.1f}%)"
    )

    # Determine the file type based on the extension
    file_mode = "rb" if bam_path.suffix == ".bam" else "r"
    output_path = bam_path.with_suffix(".filtered.bam")

    bam_file = pysam.AlignmentFile(bam_path.as_posix(), file_mode)
    output_file = pysam.AlignmentFile(output_path.as_posix(), "wb", template=bam_file)

    try:
        reads = bam_file.fetch()
        if progress_bar:
            reads = track(reads, description="Filtering BAM file")

        for read in reads:
            if predictions.get(read.query_name) is not None and predictions[read.query_name] == 1:
                continue
            output_file.write(read)

        output_file.close()
        bam_file.close()

    except Exception as e:
        logging.error(f"Error filtering BAM file: {e}")
        if output_path.exists():
            output_path.unlink()
        raise e

    if index:
        logging.info(f"Sorting {output_path}")
        sorted_output_path = output_path.with_suffix(".sorted.bam")
        pysam.sort("-o", sorted_output_path.as_posix(), output_path.as_posix())
        logging.info(f"Indexing {sorted_output_path}")
        pysam.index(sorted_output_path.as_posix())


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


class OrderCommands(TyperGroup):
    """Order commands in the order appear."""

    def list_commands(self, ctx: Context):
        """Return list of commands in the order appear."""
        return list(self.commands)  # get commands using self.commands


def version_callback(value: bool):
    """Print the version and exit."""
    if value:
        raise typer.Exit()


app = typer.Typer(
    cls=OrderCommands,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="ChimeraLM: A genomic lanuage model to identify chimera artifact.",
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


@app.command()
def predict(
    data_path: Path = typer.Argument(..., help="Path to the dataset"),
    gpus: int = typer.Option(0, "--gpus", "-g", help="Number of GPUs to use"),
    output_path: Path | None = typer.Option(None, "--output", "-o", help="Output path for predictions"),
    batch_size: int = typer.Option(12, "--batch-size", "-b", help="Batch size"),
    num_workers: int = typer.Option(0, "--workers", "-w", help="Number of workers"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    max_sample: int | None = typer.Option(None, "--max-sample", "-m", help="Maximum number of samples to process"),
    limit_predict_batches: int | None = typer.Option(None, "--limit-batches", "-l", help="Limit prediction batches"),
    progress_bar: bool = typer.Option(False, "--progress-bar", "-p", help="Show progress bar"),
    random_seed: bool = typer.Option(False, "--random-seed", help="Make the prediction not deterministic"),
):
    """Predict the given dataset using DeepChopper."""
    set_logging_level(logging.DEBUG if verbose else logging.INFO)

    if not random_seed:
        lightning.seed_everything(42, workers=True)

    tokenizer = chimeralm.data.tokenizer.load_tokenizer_from_hyena_model("hyenadna-small-32k-seqlen")
    datamodule: lightning.LightningDataModule = chimeralm.data.bam.BamDataModule(
        train_data_path="dummy.bam",
        tokenizer=tokenizer,
        predict_data_path=data_path.as_posix(),
        batch_size=batch_size,
        num_workers=num_workers,
        max_predict_samples=max_sample,
    )

    model = chimeralm.models.ChimeraLM.from_pretrained("yangliz5/chimeralm")

    if output_path is None:
        output_path = data_path.with_suffix(".predictions")

    callbacks = [chimeralm.models.callbacks.PredictionWriter(output_dir=output_path, write_interval="batch")]

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
        deterministic=not random_seed,
        logger=False,
        limit_predict_batches=limit_predict_batches,
    )

    ctx._force_start_method("spawn")
    trainer.predict(model=model, dataloaders=datamodule, return_predictions=False)
    filter_bam_by_predcition(data_path, output_path / "0", index=True)


@app.command()
def filter(
    bam_path: Path = typer.Argument(..., help="Path to the BAM file"),
    predictions_path: Path = typer.Argument(..., help="Path to the predictions file"),
    *,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
):
    """Filter the BAM file by predictions."""
    if verbose:
        set_logging_level(logging.INFO)

    filter_bam_by_predcition(bam_path, predictions_path, index=True)


if __name__ == "__main__":
    main()
