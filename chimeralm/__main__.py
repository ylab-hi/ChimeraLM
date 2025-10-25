import logging
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import lightning
import multiprocess.context as ctx
import pysam
import torch
import typer
from click import Context
from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text
from typer.core import TyperGroup

import chimeralm
from chimeralm.utils import RankedLogger

console = Console()

log = RankedLogger(__name__, rank_zero_only=True)


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
        raise ValueError(msg) from e

    return predicts


def load_predictions_from_folder(path: Path | str) -> dict[str, int]:
    """Load predictions from a folder."""
    predictions: dict[str, int] = {}
    for file in Path(path).glob("*.txt"):
        predictions.update(load_predicts(file))
    return predictions


def collect_txt_from_file(path: Path | str) -> Iterator[Path]:
    """Collect txt files from a single prediction file.

    Args:
        path: Path to the prediction file

    Yields:
        Path to the txt file

    """
    path = Path(path)
    if not path.exists():
        log.error(f"File not found: {path}")
        raise typer.Exit(1)

    yield from path.glob("*.txt")


def set_tensor_core_precision(precision="medium") -> None:
    """Set Tensor Core precision for NVIDIA GPUs."""
    # Check if using H100 or A100 and enable Tensor Core operations accordingly
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name()
        if "H100" in device_name or "A100" in device_name:
            log.info(f"Enabling {precision=} Tensor Cores for {device_name}")
            torch.set_float32_matmul_precision(precision)


def filter_bam_by_predcition(
    bam_path: Path, prediction_path: Path, *, index: bool = True, output_prediction: bool = False
) -> None:
    """Filter a BAM file by predictions.

    use parallel processing if n_jobs is greater than 1
    """
    predictions = load_predictions_from_folder(prediction_path)
    if not predictions:
        log.warning("No predictions found")
        return

    if output_prediction:
        log.info(f"Writing all predictions to {prediction_path / 'predictions.txt'}")
        with Path(prediction_path / "predictions.txt").open("w") as f:
            for name, label in predictions.items():
                f.write(f"{name}\t{label}\n")

    log.info(f"Loaded {len(predictions)} predictions from {prediction_path}")

    # summar 0 and 1 predictions
    counter = Counter(predictions.values())
    log.info(
        f"Biological: {counter.get(0, 0)} ({counter.get(0, 0) / len(predictions) * 100:.1f}%), Chimera artifact: {counter.get(1, 0)} ({counter.get(1, 0) / len(predictions) * 100:.1f}%)"
    )

    # Determine the file type based on the extension
    file_mode: Literal["rb", "r"] = "rb" if bam_path.suffix == ".bam" else "r"
    output_path = bam_path.with_suffix(".filtered.bam")

    bam_file = pysam.AlignmentFile(bam_path.as_posix(), file_mode)
    output_file = pysam.AlignmentFile(output_path.as_posix(), "wb", template=bam_file)

    try:
        reads = bam_file.fetch()
        for read in reads:
            if predictions.get(read.query_name) is not None and predictions[read.query_name] == 1:
                continue
            output_file.write(read)

        output_file.close()
        bam_file.close()

    except Exception as e:
        log.error(f"Error filtering BAM file: {e}")
        if output_path.exists():
            output_path.unlink()
        raise e

    if index:
        log.info(f"Sorting {output_path}")
        sorted_output_path = output_path.with_suffix(".sorted.bam")
        pysam.sort("-o", sorted_output_path.as_posix(), output_path.as_posix())
        log.info(f"Indexing {sorted_output_path}")
        pysam.index(sorted_output_path.as_posix())


def set_logging_level(level: int = logging.INFO):
    """Set the logging level.

    Parameters
    ----------
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
        typer.echo(f"ChimeraLM v{chimeralm.__version__}")
        raise typer.Exit()


def print_logo():
    """Print ChimeraLM pixel art logo."""
    logo_lines = [
        "",
        "",
        "    \033[38;2;160;121;185m▄\033[0m\033[38;2;145;86;186m\033[48;2;155;113;182m▀\033[0m\033[38;2;154;111;180m▄\033[0m\033[38;2;131;79;163m▄\033[0m\033[38;2;141;94;171m\033[48;2;180;149;199m▀\033[0m\033[38;2;115;56;151m▀\033[0m\033[38;2;115;56;151m▀\033[0m\033[38;2;141;94;171m\033[48;2;180;149;199m▀\033[0m\033[38;2;131;79;163m▄\033[0m\033[38;2;154;111;180m▄\033[0m\033[38;2;145;86;186m\033[48;2;155;113;182m▀\033[0m\033[38;2;160;121;185m▄\033[0m    ",
        "     \033[38;2;185;157;203m▀\033[0m\033[38;2;115;54;153m\033[48;2;110;44;151m▀\033[0m\033[38;2;145;91;183m\033[48;2;128;66;170m▀\033[0m\033[38;2;169;125;200m\033[48;2;116;52;157m▀\033[0m\033[38;2;168;122;202m\033[48;2;151;89;196m▀\033[0m\033[38;2;168;122;202m\033[48;2;151;89;196m▀\033[0m\033[38;2;169;125;200m\033[48;2;116;52;157m▀\033[0m\033[38;2;145;91;183m\033[48;2;128;66;170m▀\033[0m\033[38;2;115;54;153m\033[48;2;110;44;151m▀\033[0m\033[38;2;185;157;203m▀\033[0m     ",
        "      \033[38;2;111;46;151m\033[48;2;107;41;147m▀\033[0m\033[38;2;153;94;196m\033[48;2;140;78;185m▀\033[0m\033[38;2;151;92;194m\033[48;2;143;81;189m▀\033[0m\033[38;2;161;103;205m\033[48;2;127;64;169m▀\033[0m\033[38;2;161;103;205m\033[48;2;127;64;169m▀\033[0m\033[38;2;151;92;194m\033[48;2;143;81;189m▀\033[0m\033[38;2;153;94;196m\033[48;2;140;78;185m▀\033[0m\033[38;2;111;46;151m\033[48;2;107;41;147m▀\033[0m      ",
        "  \033[38;2;148;90;190m▀\033[0m  \033[38;2;193;169;208m▀\033[0m\033[38;2;108;43;149m▀\033[0m\033[38;2;164;107;207m\033[48;2;147;86;192m▀\033[0m\033[38;2;160;101;203m\033[48;2;161;103;204m▀\033[0m\033[38;2;116;54;155m\033[48;2;153;95;195m▀\033[0m\033[38;2;116;54;155m\033[48;2;153;95;195m▀\033[0m\033[38;2;160;101;203m\033[48;2;161;103;204m▀\033[0m\033[38;2;164;107;207m\033[48;2;147;86;192m▀\033[0m\033[38;2;108;43;149m▀\033[0m\033[38;2;193;169;208m▀\033[0m  \033[38;2;148;90;190m▀\033[0m  ",
        "  \033[38;2;183;141;213m\033[48;2;160;108;198m▀\033[0m    \033[38;2;156;109;189m▀\033[0m\033[38;2;128;64;171m\033[48;2;130;77;163m▀\033[0m\033[38;2;138;78;178m\033[48;2;111;47;151m▀\033[0m\033[38;2;138;78;178m\033[48;2;111;47;151m▀\033[0m\033[38;2;128;64;171m\033[48;2;130;77;163m▀\033[0m\033[38;2;156;109;189m▀\033[0m    \033[38;2;183;141;213m\033[48;2;160;108;198m▀\033[0m  ",
        "        \033[38;2;178;130;215m▄\033[0m\033[38;2;144;83;188m▀\033[0m\033[38;2;144;83;188m▀\033[0m\033[38;2;178;130;215m▄\033[0m",
    ]

    for line in logo_lines:
        console.print(Text.from_ansi(line))
    console.print()
    console.print("      ChimeraLM        ", style="bold cyan")
    console.print("  Genomic Language Model", style="dim cyan")


app = typer.Typer(
    cls=OrderCommands,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="ChimeraLM: A genomic lanuage model to identify chimera artifact introduced by whole genome amplification (WGA).",
)


# Add the version option to the main app
@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
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
    if ctx.invoked_subcommand is None:
        print_logo()
        console.print("\n[yellow]Run with --help to see available commands[/yellow]\n")


def determine_accelerator_and_devices(gpus: int):
    """Determine the accelerator and devices to use."""
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
    return accelerator, devices


@app.command()
def predict(
    data_path: Path = typer.Argument(..., help="Path to the dataset"),
    output_path: Path  = typer.Argument(..., "--output", "-o", help="Output path for predictions"),
    gpus: int = typer.Option(0, "--gpus", "-g", help="Number of GPUs to use"),
    batch_size: int = typer.Option(12, "--batch-size", "-b", help="Batch size"),
    num_workers: int = typer.Option(0, "--workers", "-w", help="Number of workers"),
    ckpt_path: Path | None = typer.Option(None, "--ckpt", "-c", hidden=True, help="Path to the checkpoint file"),
    *,
    random: bool = typer.Option(False, "--random", "-r", help="Make the prediction not deterministic"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
):
    """Predict the given dataset using ChimeraLM."""
    set_logging_level(logging.DEBUG if verbose else logging.INFO)
    set_tensor_core_precision()

    if not random:
        lightning.seed_everything(42, workers=True)


    tokenizer = chimeralm.data.tokenizer.load_tokenizer_from_hyena_model("hyenadna-small-32k-seqlen")
    datamodule: lightning.LightningDataModule = chimeralm.data.bam.BamDataModule(
        train_data_path=Path("dummy.bam"),
        tokenizer=tokenizer,
        predict_data_path=data_path,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    callbacks = [
        chimeralm.models.callbacks.PredictionWriter(output_dir=output_path, write_interval="batch"),
    ]

    if ckpt_path is not None:
        log.info(f"Loading model from {ckpt_path}")
        model = chimeralm.models.ChimeraLM.new()
        callbacks.extend(
            [
                lightning.pytorch.callbacks.RichProgressBar(),
                lightning.pytorch.callbacks.ModelCheckpoint(
                    dirpath=output_path / "checkpoints",
                    filename="epoch_{epoch:03d}_f1_{val/f1:.4f}",
                    monitor="val/f1",
                    mode="max",
                    save_last=True,
                    auto_insert_metric_name=False,
                ),
                lightning.pytorch.callbacks.EarlyStopping(monitor="val/f1", patience=40, mode="max"),
                lightning.pytorch.callbacks.ModelSummary(max_depth=1),
            ]
        )
    else:
        log.info("Loading model from Hugging Face")
        model = chimeralm.models.ChimeraLM.from_pretrained("yangliz5/chimeralm")

    if output_path is None:
        output_path = data_path.with_suffix(".predictions")
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    accelerator, devices = determine_accelerator_and_devices(gpus)
    trainer = lightning.pytorch.trainer.Trainer(
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        deterministic=not random,
        logger=False,
    )

    ctx._force_start_method("spawn")
    trainer.predict(model=model, dataloaders=datamodule, return_predictions=False, ckpt_path=ckpt_path)
    log.info(f"Predictions saved to {output_path}")
    log.info(f"Filtering {data_path} by predictions from {output_path}")


@app.command()
def filter(
    bam_path: Path = typer.Argument(..., help="Path to the BAM file"),
    predictions_path: Path = typer.Argument(..., help="Path to the predictions file"),
    *,
    output_prediction: bool = typer.Option(False, "--output-prediction", "-p", help="write summary of the predictions"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
):
    """Filter the BAM file by predictions."""
    set_logging_level(logging.DEBUG if verbose else logging.INFO)
    log.info(f"Filtering {bam_path} by predictions from {predictions_path}")
    filter_bam_by_predcition(bam_path, predictions_path, index=True, output_prediction=output_prediction)


@app.command()
def web():
    """Launch the web interface."""
    chimeralm.ui.main()


if __name__ == "__main__":
    app()
