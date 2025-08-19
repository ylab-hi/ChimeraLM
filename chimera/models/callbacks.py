from pathlib import Path

import torch
from lightning.pytorch.callbacks import BasePredictionWriter


class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval="epoch"):
        super().__init__(write_interval)
        self.output_dir = Path(output_dir)

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        folder = self.output_dir / str(dataloader_idx)
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)

        save_prediction = {
            "prediction": prediction[0].cpu(),
            "labels": prediction[1].to(torch.int64).cpu(),
            "id": batch["id"].to(torch.int64).cpu(),
        }

        torch.save(save_prediction, folder / f"{trainer.global_rank}_{batch_idx}.pt")

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # WARN: This is a simple implementation that saves all predictions in a single file
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=False, exist_ok=True)

        torch.save(predictions, self.output_dir / "predictions.pt")

def resume_read_name(bytes_data: torch.Tensor | list[int]) -> str:
    """
    Convert bytes data to a read name string.
    
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
            return ""
        
        # More efficient string building
        read_name_bytes = bytes_data[1:1 + read_name_length]
        return ''.join(chr(b) for b in read_name_bytes if 32 <= b <= 126)
    except (IndexError, TypeError, ValueError) as e:
        logger.warning(f"Error processing read name: {e}")
        return ""

class PredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval="batch"):
        super().__init__(write_interval)
        self.output_dir = Path(output_dir)

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        folder = self.output_dir / str(dataloader_idx)
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
        
        prediction = prediction[0].argmax(dim=1).cpu()
        read_names = [resume_read_name(batch["id"][i]) for i in range(len(prediction))]

        with open(folder / f"{trainer.global_rank}_{batch_idx}.txt", "w") as f:
            for read_name, pred in zip(read_names, prediction, strict=True):
                f.write(f"{read_name}\t{pred}\n")