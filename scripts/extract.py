from pathlib import Path
from typing import Dict, TextIO
import logging

import pyfastx
import typer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer()


def load_read_assignments(file_path: str) -> Dict[str, bool]:
    """Load read assignments from a file and return a dictionary mapping read names to boolean targets."""
    assignments = {}
    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                
                parts = line.split()
                if len(parts) != 2:
                    logger.warning(f"Skipping malformed line {line_num} in {file_path}: {line}")
                    continue
                
                read_name, target_str = parts
                # Convert string to boolean, handling various formats
                target = target_str.lower() in ('true', '1', 't', 'yes')
                assignments[read_name] = target
                
        logger.info(f"Loaded {len(assignments)} read assignments from {file_path}")
        return assignments
        
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        raise


def write_fastq_entry(file_handle: TextIO, read_name: str, target: bool, 
                     sequence: str, quality: str) -> None:
    """Write a FASTQ entry to the given file handle."""
    target_value = 1 if target else 0
    file_handle.write(f"@{read_name}|{target_value}\n{sequence}\n+\n{quality}\n")


@app.command()
def extract(
    fq_file: str = typer.Argument(..., help="Input FASTQ file"),
    train_file: str = typer.Argument(..., help="Training data file"),
    validation_file: str = typer.Argument(..., help="Validation data file"),
    test_file: str = typer.Argument(..., help="Test data file"),
    output_dir: str = typer.Option(".", help="Output directory for FASTQ files")
):
    """Extract chimeric reads from a FASTQ file and rename reads with target labels."""
    
    # Validate input files
    input_files = [fq_file, train_file, validation_file, test_file]
    for file_path in input_files:
        if not Path(file_path).exists():
            raise typer.BadParameter(f"Input file does not exist: {file_path}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load read assignments
    logger.info("Loading read assignments...")
    train_reads = load_read_assignments(train_file)
    validation_reads = load_read_assignments(validation_file)
    test_reads = load_read_assignments(test_file)
    
    # Check for overlapping read names
    train_set = set(train_reads.keys())
    validation_set = set(validation_reads.keys())
    test_set = set(test_reads.keys())
    
    overlaps = train_set & validation_set & test_set
    if overlaps:
        logger.warning(f"Found {len(overlaps)} overlapping read names across datasets")
    
    # Prepare output file paths
    train_output = output_path / f"{Path(train_file).stem}.fastq"
    validation_output = output_path / f"{Path(validation_file).stem}.fastq"
    test_output = output_path / f"{Path(test_file).stem}.fastq"
    
    # Counters for statistics
    stats = {
        'train': 0,
        'validation': 0,
        'test': 0,
        'unassigned': 0,
        'total_processed': 0
    }
    
    logger.info("Processing FASTQ file...")
    
    try:
        fq = pyfastx.Fastx(fq_file)
        with train_output.open('w') as train_fq, \
             validation_output.open('w') as validation_fq, \
             test_output.open('w') as test_fq:
            
            for read_name, sequence, quality in fq:
                stats['total_processed'] += 1
                
                if read_name in train_reads:
                    write_fastq_entry(train_fq, read_name, train_reads[read_name], sequence, quality)
                    stats['train'] += 1
                elif read_name in validation_reads:
                    write_fastq_entry(validation_fq, read_name, validation_reads[read_name], sequence, quality)
                    stats['validation'] += 1
                elif read_name in test_reads:
                    write_fastq_entry(test_fq, read_name, test_reads[read_name], sequence, quality)
                    stats['test'] += 1
                else:
                    stats['unassigned'] += 1
                    
                # Log progress every 10000 reads
                if stats['total_processed'] % 10000 == 0:
                    logger.info(f"Processed {stats['total_processed']} reads...")
    
    except Exception as e:
        logger.error(f"Error processing FASTQ file: {e}")
        raise
    
    # Print final statistics
    logger.info("Extraction completed successfully!")
    logger.info(f"Total reads processed: {stats['total_processed']}")
    logger.info(f"Training reads: {stats['train']}")
    logger.info(f"Validation reads: {stats['validation']}")
    logger.info(f"Test reads: {stats['test']}")
    logger.info(f"Unassigned reads: {stats['unassigned']}")
    logger.info(f"Output files:")
    logger.info(f"  Training: {train_output}")
    logger.info(f"  Validation: {validation_output}")
    logger.info(f"  Test: {test_output}")


if __name__ == "__main__":
    app()
