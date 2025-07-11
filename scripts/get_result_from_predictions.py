import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator
import typer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


@dataclass
class Predict:
    """Data class representing a prediction result."""
    name: str
    label: int
    sv: str | None = None


def collect_predict_from_file(path: Path | str) -> Iterator[Predict]:
    """
    Collect predictions from a single PyTorch file.
    
    Args:
        path: Path to the prediction file
        
    Yields:
        Predict objects
    """
    try:
        path = Path(path)
        if not path.exists():
            logger.error(f"File not found: {path}")
            return
            
        predicts = torch.load(path, weights_only=True, map_location='cpu')
        
        if "id" not in predicts or "prediction" not in predicts:
            logger.error(f"Invalid prediction file format: {path}")
            return
            
        read_names = [resume_read_name(id_data) for id_data in predicts["id"]]
        labels = predicts["prediction"].argmax(dim=1).tolist()
        
        for name, label in zip(read_names, labels):
            yield Predict(name=name, label=label)
            
    except Exception as e:
        logger.error(f"Error processing file {path}: {e}")


def collect_predict_from_folder(folder: Path | str) -> Iterator[Predict]:
    """
    Collect predictions from all .pt files in a folder.
    
    Args:
        folder: Path to the folder containing prediction files
        
    Yields:
        Predict objects from all files
    """
    try:
        folder = Path(folder)
        if not folder.exists():
            logger.error(f"Folder not found: {folder}")
            return
            
        if not folder.is_dir():
            logger.error(f"Path is not a directory: {folder}")
            return
            
        pt_files = list(folder.glob("*.pt"))
        if not pt_files:
            logger.warning(f"No .pt files found in {folder}")
            return
            
        for file in pt_files:
            logger.debug(f"Processing file: {file}")
            yield from collect_predict_from_file(file)
            
    except Exception as e:
        logger.error(f"Error processing folder {folder}: {e}")


def write_predicts(predicts: Iterator[Predict], path: Path | str) -> tuple[int, int]:
    """
    Write predictions to a file and return summary statistics.
    
    Args:
        predicts: Iterator of Predict objects
        path: Output file path
        
    Returns:
        Tuple of (total_count, label_1_count)
    """
    total = 0
    number_label_1 = 0
    
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding='utf-8') as f:
            for predict in predicts:
                total += 1
                if predict.label == 1:
                    number_label_1 += 1
                f.write(f"{predict.name}\t{predict.label}\n")
                
        logger.info(f"Wrote {total} predictions to {path}")
        logger.info(f"Label 1 count: {number_label_1}/{total} ({number_label_1/total*100:.1f}%)")
        
    except Exception as e:
        logger.error(f"Error writing to {path}: {e}")
        
    return total, number_label_1


def load_predicts(path: Path | str) -> list[Predict]:
    """
    Load predictions from a text file.
    
    Args:
        path: Path to the input file
        
    Returns:
        List of Predict objects
    """
    predicts = []
    try:
        path = Path(path)
        if not path.exists():
            logger.error(f"File not found: {path}")
            return predicts
            
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    parts = line.split("\t")
                    if len(parts) != 2:
                        logger.warning(f"Invalid line format at line {line_num}: {line}")
                        continue
                        
                    name, label_str = parts
                    label = int(label_str)
                    predicts.append(Predict(name=name, label=label))
                    
                except ValueError as e:
                    logger.warning(f"Invalid label at line {line_num}: {e}")
                    continue
                    
    except Exception as e:
        logger.error(f"Error reading file {path}: {e}")
        
    return predicts


app = typer.Typer()


@app.command()
def get_result_from_predictions(
    input_path: Path = typer.Argument(..., help="Path to the input folder containing prediction files"),
    output_path: Path = typer.Argument(..., help="Path to the output file"),
):
    """
    Process prediction files and write results to output file.
    """
    try:
        if not input_path.exists():
            logger.error(f"Input path does not exist: {input_path}")
            raise typer.Exit(1)
            
        predicts = collect_predict_from_folder(input_path)
        total, label_1_count = write_predicts(predicts, output_path)
        
        if total == 0:
            logger.warning("No predictions were processed")
        else:
            logger.info(f"Successfully processed {total} predictions")
            
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()