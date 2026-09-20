from pathlib import Path
from typing import Any

import torch
from lightning.pytorch.callbacks import BasePredictionWriter

from chimeralm.utils import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)

# Printable ASCII range kept when decoding a stored read name.
_PRINTABLE_ASCII_MIN = 32
_PRINTABLE_ASCII_MAX = 126


class CustomWriter(BasePredictionWriter):
    """Prediction writer that stores raw prediction tensors to a folder."""

    def __init__(self, output_dir, write_interval="epoch"):
        """Initialize the writer with the folder that receives the predictions.

        Args:
            output_dir: Folder the tensors are saved to. Created on first write.
            write_interval: Lightning write interval, ``"batch"`` or ``"epoch"``.

        """
        super().__init__(write_interval)
        self.output_dir = Path(output_dir)

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        """Save one batch of predictions as `{rank}_{batch_idx}.pt`."""
        folder = self.output_dir
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)

        save_prediction = {
            "prediction": prediction[0].cpu(),
            "labels": prediction[1].to(torch.int64).cpu(),
            "id": batch["id"].to(torch.int64).cpu(),
        }

        torch.save(save_prediction, folder / f"{trainer.global_rank}_{batch_idx}.pt")

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        """Save all epoch predictions into a single `predictions.pt` file."""
        # WARN: This is a simple implementation that saves all predictions in a single file
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=False, exist_ok=True)

        torch.save(predictions, self.output_dir / "predictions.pt")


def resume_read_name(bytes_data: torch.Tensor | list[int]) -> str:
    """Convert bytes data to a read name string.

    Args:
        bytes_data: Tensor or list of integers representing bytes

    Returns:
        Extracted read name string

    """
    # Convert bytes to string
    if isinstance(bytes_data, torch.Tensor):
        if bytes_data.numel() == 0:
            return ""
        bytes_data = bytes_data.tolist()
    elif not bytes_data:
        return ""

    try:
        read_name_length = bytes_data[0]
        if read_name_length <= 0 or read_name_length >= len(bytes_data):
            raise ValueError("Invalid read name length")
        # More efficient string building
        read_name_bytes = bytes_data[1 : 1 + read_name_length]
        return "".join(chr(b) for b in read_name_bytes if _PRINTABLE_ASCII_MIN <= b <= _PRINTABLE_ASCII_MAX)
    except (IndexError, TypeError, ValueError) as e:
        raise ValueError("Invalid read name data") from e


class PredictionWriter(BasePredictionWriter):
    """Optimized prediction writer with improved error handling and performance."""

    def __init__(self, output_dir: str | Path, write_interval: str = "batch") -> None:
        """Initialize the prediction writer.

        Args:
            output_dir: Directory to write predictions to
            write_interval: When to write predictions (batch or epoch)

        """
        super().__init__(write_interval)
        self.output_dir = Path(output_dir)

    def write_on_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        prediction: Any,
        batch_indices: Any,
        batch: dict[str, Any],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Optimized batch writing with better error handling and performance improvements."""
        try:
            # Input validation
            if not prediction or len(prediction) == 0:
                logger.warning(f"Empty prediction for batch {batch_idx}, dataloader {dataloader_idx}")
                return

            if "id" not in batch:
                logger.error(f"Missing 'id' key in batch {batch_idx}, dataloader {dataloader_idx}")
                return

            # Get predictions - handle both single tensor and tuple cases
            pred_tensor = prediction[0] if isinstance(prediction, list | tuple) else prediction
            if pred_tensor is None or pred_tensor.numel() == 0:
                logger.warning(f"Empty prediction tensor for batch {batch_idx}")
                return

            # Optimize tensor operations - do argmax and CPU transfer in one go
            predictions_cpu = pred_tensor.argmax(dim=1).cpu()
            batch_ids = batch["id"]

            # Validate tensor sizes match
            if len(predictions_cpu) != len(batch_ids):
                logger.error(
                    f"Size mismatch: predictions={len(predictions_cpu)}, "
                    f"batch_ids={len(batch_ids)} for batch {batch_idx}"
                )
                return

            # Batch process read names with error handling
            read_names = []
            for i, batch_id in enumerate(batch_ids):
                try:
                    read_name = resume_read_name(batch_id)
                    if not read_name:  # Handle empty read names
                        read_name = f"unknown_read_{i}"
                        logger.warning(f"Empty read name for index {i} in batch {batch_idx}")
                    read_names.append(read_name)
                except (IndexError, TypeError, ValueError) as e:
                    logger.error(f"Error processing read name at index {i}: {e}")
                    read_names.append(f"error_read_{i}")

            if not self.output_dir.exists():
                self.output_dir.mkdir(parents=False, exist_ok=True)
            # Write results with buffered I/O for better performance
            output_file = self.output_dir / f"{trainer.global_rank}_{batch_idx}.txt"
            try:
                # Use list comprehension and join for more efficient string building
                lines = [
                    f"{read_name}\t{pred.item()}\n" for read_name, pred in zip(read_names, predictions_cpu, strict=True)
                ]

                with output_file.open("w") as f:
                    f.writelines(lines)

            except OSError as e:
                logger.error(f"Failed to write predictions to {output_file}: {e}")
            except (RuntimeError, TypeError, ValueError) as e:
                logger.error(f"Unexpected error writing batch {batch_idx}: {e}")

        # Last-resort boundary: a failing writer must not abort the prediction run.
        except Exception as e:  # noqa: BLE001
            logger.error(f"Critical error in write_on_batch_end for batch {batch_idx}: {e}")
