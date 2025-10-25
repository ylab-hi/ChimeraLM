# Utils API Reference

Python API reference for ChimeraLM utility functions and helper modules.

## Overview

ChimeraLM's utility modules provide:

- **Logging**: Structured logging with Rich integration
- **Configuration**: Hydra configuration helpers
- **Metrics**: Evaluation metrics and reporting
- **File operations**: BAM/SAM file utilities

## Module Structure

```
chimeralm.utils/
├── logging.py      # Logging utilities
├── config.py       # Configuration helpers
├── metrics.py      # Evaluation metrics
└── io.py           # File I/O utilities
```

---

## Logging Utilities


### Usage

The `RankedLogger` ensures logging only happens on the main process in distributed training.

```python
from chimeralm.utils.logging import RankedLogger

# Create logger
log = RankedLogger(__name__, rank_zero_only=True)

# Log messages (only on rank 0)
log.info("Training started")
log.warning("High memory usage detected")
log.error("Training failed")
```

### Methods

#### `info(msg)`

Log informational message.

**Parameters:**
- `msg` (str): Message to log

**Example:**
```python
log.info(f"Epoch {epoch}: loss={loss:.4f}")
```

#### `warning(msg)`

Log warning message.

**Parameters:**
- `msg` (str): Warning message

**Example:**
```python
log.warning("GPU memory is 95% full")
```

#### `error(msg)`

Log error message.

**Parameters:**
- `msg` (str): Error message

**Example:**
```python
log.error(f"Failed to load checkpoint: {e}")
```

#### `debug(msg)`

Log debug message.

**Parameters:**
- `msg` (str): Debug message

**Example:**
```python
log.debug(f"Batch shape: {batch['input_ids'].shape}")
```

---

## Configuration Helpers


### Usage

Resolve paths in Hydra configurations.

```python
from chimeralm.utils.config import resolve_paths
from omegaconf import DictConfig

# Configuration with relative paths
config = DictConfig({
    "data": {
        "train_path": "data/train.bam",
        "output_dir": "results/"
    }
})

# Resolve to absolute paths
resolve_paths(config, base_dir="/project/chimera")

print(config.data.train_path)   # /project/chimera/data/train.bam
print(config.data.output_dir)   # /project/chimera/results/
```

---

## Evaluation Metrics


### Usage

Compute classification metrics (accuracy, precision, recall, F1).

```python
from chimeralm.utils.metrics import compute_classification_metrics
import torch

# Predictions and ground truth
preds = torch.tensor([0, 1, 0, 1, 0])
targets = torch.tensor([0, 1, 0, 0, 1])

# Compute metrics
metrics = compute_classification_metrics(preds, targets)

print(metrics)
```

**Output:**
```python
{
    'accuracy': 0.6,
    'precision': 0.5,
    'recall': 0.5,
    'f1': 0.5
}
```

### Parameters

#### `preds` (torch.Tensor)

Predicted labels (0 or 1).

#### `targets` (torch.Tensor)

Ground truth labels (0 or 1).

### Returns

Dictionary with metrics:
- `accuracy`: Overall accuracy
- `precision`: Precision score
- `recall`: Recall score
- `f1`: F1 score

---

## File I/O Utilities

### Reading BAM Files


```python
from chimeralm.utils.io import read_bam_reads

# Read all reads with SA tags
reads = read_bam_reads("input.bam", filter_sa_tags=True)

print(f"Found {len(reads)} chimeric candidates")

# Access read data
for read in reads[:5]:
    print(f"Name: {read.query_name}")
    print(f"Sequence: {read.query_sequence[:50]}...")
    print(f"Has SA tag: {read.has_tag('SA')}")
```

### Writing Predictions


```python
from chimeralm.utils.io import write_predictions

# Predictions dictionary
predictions = {
    "read1": 0,  # Biological
    "read2": 1,  # Chimeric
    "read3": 0   # Biological
}

# Write to file
write_predictions(
    predictions,
    output_path="predictions.txt",
    format="tsv"  # Tab-separated
)
```

**Output file (predictions.txt):**
```text
read1	0
read2	1
read3	0
```

### Filtering BAM Files


```python
from chimeralm.utils.io import filter_bam_by_predictions

# Filter BAM file
filter_bam_by_predictions(
    input_bam="input.bam",
    predictions_file="predictions.txt",
    output_bam="filtered.bam",
    keep_label=0  # Keep biological reads (label 0)
)

print("Filtering complete: filtered.bam")
```

---

## Complete Example: Custom Evaluation

```python
from chimeralm.models.lm import ChimeraLM
from chimeralm.data.bam import BamDataModule
from chimeralm.utils.metrics import compute_classification_metrics
from chimeralm.utils.logging import RankedLogger
import torch

# Setup logging
log = RankedLogger(__name__)

# Load model
log.info("Loading model...")
model = ChimeraLM.from_pretrained("yangliz5/chimeralm")
model.eval()

# Load data
log.info("Loading data...")
data_module = BamDataModule(
    train_data_path="test.bam",
    batch_size=32
)
data_module.setup("test")
test_loader = data_module.test_dataloader()

# Run evaluation
log.info("Running evaluation...")
all_preds = []
all_targets = []

with torch.no_grad():
    for batch in test_loader:
        # Forward pass
        logits = model(batch["input_ids"])
        preds = torch.argmax(logits, dim=-1)

        # Collect predictions
        all_preds.append(preds)
        all_targets.append(batch["labels"])

# Concatenate all batches
all_preds = torch.cat(all_preds)
all_targets = torch.cat(all_targets)

# Compute metrics
metrics = compute_classification_metrics(all_preds, all_targets)

# Log results
log.info("Evaluation Results:")
log.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
log.info(f"  Precision: {metrics['precision']:.4f}")
log.info(f"  Recall:    {metrics['recall']:.4f}")
log.info(f"  F1 Score:  {metrics['f1']:.4f}")
```

**Output:**
```text
Loading model...
Loading data...
Running evaluation...
Evaluation Results:
  Accuracy:  0.9234
  Precision: 0.9156
  Recall:    0.9312
  F1 Score:  0.9234
```

---

## Complete Example: Custom Data Processing

```python
from chimeralm.utils.io import read_bam_reads, write_predictions
from chimeralm.data.tokenizer import DNATokenizer
from chimeralm.models.lm import ChimeraLM
import torch

# 1. Read BAM file
print("Reading BAM file...")
reads = read_bam_reads("input.bam", filter_sa_tags=True)
print(f"Found {len(reads)} reads with SA tags")

# 2. Extract sequences
sequences = [read.query_sequence for read in reads]
read_names = [read.query_name for read in reads]

# 3. Tokenize sequences
tokenizer = DNATokenizer()
token_ids = [tokenizer.encode(seq) for seq in sequences]

# Pad to same length
from torch.nn.utils.rnn import pad_sequence
token_tensors = [torch.tensor(ids) for ids in token_ids]
padded_tokens = pad_sequence(token_tensors, batch_first=True, padding_value=5)

# 4. Run predictions
model = ChimeraLM.from_pretrained("yangliz5/chimeralm")
model.eval()

with torch.no_grad():
    logits = model(padded_tokens)
    predictions = torch.argmax(logits, dim=-1)

# 5. Create predictions dictionary
pred_dict = {
    name: label.item()
    for name, label in zip(read_names, predictions)
}

# 6. Write predictions
write_predictions(pred_dict, "predictions.txt")
print(f"Wrote {len(pred_dict)} predictions to predictions.txt")

# 7. Count results
biological = sum(1 for label in predictions if label == 0)
chimeric = sum(1 for label in predictions if label == 1)
print(f"Biological: {biological}, Chimeric: {chimeric}")
```

---

## Tensor Core Optimization


Enable tensor core optimization for H100/A100 GPUs.

```python
from chimeralm.utils.config import enable_tensor_cores
import torch

# Enable tensor cores
enable_tensor_cores()

# Check if enabled
print(f"Tensor cores: {torch.get_float32_matmul_precision()}")
```

**Output:**
```text
Tensor cores: medium
```

This sets `torch.set_float32_matmul_precision("medium")` for faster training on modern GPUs.

---

## Rich Console Output


Access the Rich console for formatted output.

```python
from chimeralm.utils.logging import console

# Print with formatting
console.print("[bold green]Training complete![/bold green]")
console.print("[yellow]Warning:[/yellow] High memory usage")

# Print tables
from rich.table import Table

table = Table(title="Evaluation Results")
table.add_column("Metric", style="cyan")
table.add_column("Value", style="magenta")

table.add_row("Accuracy", "0.9234")
table.add_row("Precision", "0.9156")
table.add_row("Recall", "0.9312")

console.print(table)
```

---

## Progress Bars

```python
from chimeralm.utils.logging import console
from rich.progress import Progress

with Progress(console=console) as progress:
    task = progress.add_task("[cyan]Processing reads...", total=1000)

    for i in range(1000):
        # Do work
        process_read(i)

        # Update progress
        progress.update(task, advance=1)
```

---

## See Also

- [Models API Reference](models.md) - Model architecture
- [Data API Reference](data.md) - Data loading
- [CLI Commands](cli.md) - Command-line interface
- [Architecture Overview](../architecture/overview.md) - System design
