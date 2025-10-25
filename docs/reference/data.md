# Data API Reference

Python API reference for ChimeraLM data loading, tokenization, and preprocessing.

## Overview

ChimeraLM's data pipeline consists of:

- **Data modules**: PyTorch Lightning DataModules for BAM and FASTQ files
- **Tokenizers**: DNA sequence tokenization (character-level)
- **Data collators**: Batching and padding for efficient training
- **Dataset classes**: HuggingFace Datasets integration

## Module Structure

```
chimeralm.data/
├── bam.py         # BAM file data module
├── fq.py          # FASTQ file data module
├── tokenizer.py   # DNA sequence tokenization
└── collator.py    # Data collator for batching
```

---

## BAM Data Module

### Usage

The `BamDataModule` handles loading and preprocessing of BAM files for training and inference.

```python
from chimeralm.data.bam import BamDataModule

# Create data module
data_module = BamDataModule(
    train_data_path="train.bam",
    val_data_path="val.bam",      # Optional
    test_data_path="test.bam",    # Optional
    batch_size=32,
    num_workers=4,
    max_length=1024
)

# Setup data
data_module.setup(stage="fit")

# Get dataloaders
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()
```

### Parameters

#### `train_data_path` (str, required)

Path to training BAM file. If `val_data_path` and `test_data_path` are not provided, this file will be automatically split according to `train_val_test_split`.

**Example:**
```python
data_module = BamDataModule(train_data_path="data/train.bam")
```

#### `val_data_path` (str, optional)

Path to validation BAM file.

**Example:**
```python
data_module = BamDataModule(
    train_data_path="data/train.bam",
    val_data_path="data/val.bam"
)
```

#### `test_data_path` (str, optional)

Path to test BAM file.

#### `batch_size` (int, default: 12)

Number of sequences per batch.

**Example:**
```python
# Small batch for limited memory
data_module = BamDataModule(
    train_data_path="train.bam",
    batch_size=8
)

# Large batch for high-memory GPU
data_module = BamDataModule(
    train_data_path="train.bam",
    batch_size=64
)
```

#### `num_workers` (int, default: 0)

Number of worker processes for data loading.

**Recommendations:**
- CPU training: 4-8 workers
- GPU training: 2-4 workers
- Single-threaded: 0 (default)

**Example:**
```python
data_module = BamDataModule(
    train_data_path="train.bam",
    num_workers=4
)
```

#### `max_length` (int, default: 1024)

Maximum sequence length in tokens. Longer sequences are truncated.

**Example:**
```python
# Short sequences
data_module = BamDataModule(
    train_data_path="train.bam",
    max_length=512
)

# Long sequences
data_module = BamDataModule(
    train_data_path="train.bam",
    max_length=2048
)
```

#### `train_val_test_split` (list, default: [0.7, 0.2, 0.1])

Train/validation/test split ratios when using auto-split (i.e., when `val_data_path` and `test_data_path` are None).

**Example:**
```python
# 80% train, 15% val, 5% test
data_module = BamDataModule(
    train_data_path="all_data.bam",
    train_val_test_split=[0.8, 0.15, 0.05]
)
```

#### `max_sample` (int, optional)

Maximum number of samples to load. Useful for quick testing or processing subsets.

**Example:**
```python
# Load only first 1000 reads
data_module = BamDataModule(
    train_data_path="train.bam",
    max_sample=1000
)
```

### Methods

#### `setup(stage=None)`

Prepare data for training, validation, or testing.

**Parameters:**
- `stage` (str): One of `"fit"`, `"validate"`, `"test"`, or `"predict"`

**Example:**
```python
# Setup for training and validation
data_module.setup(stage="fit")

# Setup for testing
data_module.setup(stage="test")

# Setup for prediction
data_module.setup(stage="predict")
```

#### `train_dataloader()`

Returns the training DataLoader.

**Returns:**
- `DataLoader`: Training data loader

**Example:**
```python
data_module.setup("fit")
train_loader = data_module.train_dataloader()

for batch in train_loader:
    input_ids = batch["input_ids"]  # [batch_size, max_length]
    labels = batch["labels"]        # [batch_size]
    break
```

#### `val_dataloader()`

Returns the validation DataLoader.

**Returns:**
- `DataLoader`: Validation data loader

#### `test_dataloader()`

Returns the test DataLoader.

**Returns:**
- `DataLoader`: Test data loader

#### `predict_dataloader()`

Returns the prediction DataLoader.

**Returns:**
- `DataLoader`: Prediction data loader

### Data Format

Each batch is a dictionary with:

```python
{
    "input_ids": torch.Tensor,     # Shape: [batch_size, max_length]
    "labels": torch.Tensor,        # Shape: [batch_size]
    "attention_mask": torch.Tensor # Shape: [batch_size, max_length]
}
```

**Example:**
```python
batch = next(iter(train_loader))

print(batch["input_ids"].shape)      # torch.Size([32, 1024])
print(batch["labels"].shape)         # torch.Size([32])
print(batch["attention_mask"].shape) # torch.Size([32, 1024])

# Label values
# 0: Biological read
# 1: Chimeric read
```

---

## FASTQ Data Module

### Usage

The `FqDataModule` handles loading and preprocessing of FASTQ files.

```python
from chimeralm.data.fq import FqDataModule

# Create data module
data_module = FqDataModule(
    train_data_path="train.fastq",
    val_data_path="val.fastq",
    batch_size=32,
    num_workers=4
)

data_module.setup("fit")
train_loader = data_module.train_dataloader()
```

### Parameters

Similar to `BamDataModule`:
- `train_data_path` (str, required)
- `val_data_path` (str, optional)
- `test_data_path` (str, optional)
- `batch_size` (int, default: 12)
- `num_workers` (int, default: 0)
- `max_length` (int, default: 1024)

---

## DNA Tokenizer

### Usage

The `DNATokenizer` converts DNA sequences (strings of A, C, G, T, N) into integer token IDs.

```python
from chimeralm.data.tokenizer import DNATokenizer

# Create tokenizer
tokenizer = DNATokenizer()

# Tokenize sequence
sequence = "ACGTACGTNNACGT"
token_ids = tokenizer.encode(sequence)

print(token_ids)  # [1, 2, 3, 4, 1, 2, 3, 4, 0, 0, 1, 2, 3, 4]

# Decode back to sequence
decoded = tokenizer.decode(token_ids)
print(decoded)  # "ACGTACGTNNACGT"
```

### Token Mapping

| Token | ID |
|-------|------|
| N (or unknown) | 0 |
| A | 1 |
| C | 2 |
| G | 3 |
| T | 4 |
| [PAD] | 5 |
| [CLS] | 6 |
| [SEP] | 7 |

### Methods

#### `encode(sequence)`

Convert DNA sequence string to token IDs.

**Parameters:**
- `sequence` (str): DNA sequence (A, C, G, T, N)

**Returns:**
- `List[int]`: Token IDs

**Example:**
```python
tokenizer = DNATokenizer()

# Simple sequence
seq = "ACGT"
tokens = tokenizer.encode(seq)
print(tokens)  # [1, 2, 3, 4]

# With unknown bases
seq = "ACGTN"
tokens = tokenizer.encode(seq)
print(tokens)  # [1, 2, 3, 4, 0]

# Lowercase is supported
seq = "acgt"
tokens = tokenizer.encode(seq)
print(tokens)  # [1, 2, 3, 4]
```

#### `decode(token_ids)`

Convert token IDs back to DNA sequence string.

**Parameters:**
- `token_ids` (List[int]): Token IDs

**Returns:**
- `str`: DNA sequence

**Example:**
```python
tokenizer = DNATokenizer()

tokens = [1, 2, 3, 4, 0]
sequence = tokenizer.decode(tokens)
print(sequence)  # "ACGTN"
```

#### `batch_encode(sequences)`

Encode multiple sequences.

**Parameters:**
- `sequences` (List[str]): List of DNA sequences

**Returns:**
- `List[List[int]]`: List of token ID lists

**Example:**
```python
tokenizer = DNATokenizer()

sequences = ["ACGT", "GGCC", "TTAA"]
batch_tokens = tokenizer.batch_encode(sequences)

for seq, tokens in zip(sequences, batch_tokens):
    print(f"{seq}: {tokens}")

# Output:
# ACGT: [1, 2, 3, 4]
# GGCC: [3, 3, 2, 2]
# TTAA: [4, 4, 1, 1]
```

---

## Data Collator

### Usage

The `DataCollator` handles batching and padding of sequences.

```python
from chimeralm.data.collator import DataCollator

# Create collator
collator = DataCollator(
    tokenizer=tokenizer,
    max_length=1024,
    padding="max_length"
)

# Use with DataLoader
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=collator
)
```

### Parameters

#### `tokenizer` (DNATokenizer, required)

Tokenizer instance for encoding sequences.

#### `max_length` (int, default: 1024)

Maximum sequence length. Sequences are truncated if longer.

#### `padding` (str, default: "max_length")

Padding strategy:
- `"max_length"`: Pad all sequences to `max_length`
- `"longest"`: Pad to longest sequence in batch
- `False`: No padding

**Example:**
```python
# Pad to max_length (1024)
collator = DataCollator(
    tokenizer=tokenizer,
    max_length=1024,
    padding="max_length"
)

# Pad to longest in batch (variable length)
collator = DataCollator(
    tokenizer=tokenizer,
    padding="longest"
)

# No padding
collator = DataCollator(
    tokenizer=tokenizer,
    padding=False
)
```

---

## Complete Example: Custom Data Loading

```python
from chimeralm.data.tokenizer import DNATokenizer
from chimeralm.data.collator import DataCollator
from torch.utils.data import Dataset, DataLoader
import torch

# 1. Create custom dataset
class CustomDNADataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            "sequence": self.sequences[idx],
            "label": self.labels[idx]
        }

# 2. Prepare data
sequences = [
    "ACGTACGTACGT",
    "GGCCGGCCGGCC",
    "TTAATTAATTAA"
]
labels = [0, 1, 0]  # 0=biological, 1=chimeric

dataset = CustomDNADataset(sequences, labels)

# 3. Create tokenizer and collator
tokenizer = DNATokenizer()
collator = DataCollator(
    tokenizer=tokenizer,
    max_length=512,
    padding="max_length"
)

# 4. Create data loader
loader = DataLoader(
    dataset,
    batch_size=2,
    collate_fn=collator
)

# 5. Iterate through batches
for batch in loader:
    input_ids = batch["input_ids"]        # [2, 512]
    labels = batch["labels"]              # [2]
    attention_mask = batch["attention_mask"]  # [2, 512]

    print(f"Input shape: {input_ids.shape}")
    print(f"Labels: {labels}")
    break
```

**Output:**
```text
Input shape: torch.Size([2, 512])
Labels: tensor([0, 1])
```

---

## Complete Example: BAM File Loading

```python
from chimeralm.data.bam import BamDataModule
import lightning as L

# 1. Create data module with auto-split
data_module = BamDataModule(
    train_data_path="all_labeled_data.bam",
    batch_size=32,
    num_workers=4,
    max_length=1024,
    train_val_test_split=[0.8, 0.1, 0.1]  # 80% train, 10% val, 10% test
)

# 2. Setup for training
data_module.setup(stage="fit")

# 3. Get dataloaders
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()

# 4. Check data
batch = next(iter(train_loader))
print(f"Batch size: {batch['input_ids'].shape[0]}")
print(f"Sequence length: {batch['input_ids'].shape[1]}")
print(f"Labels: {batch['labels']}")

# Output:
# Batch size: 32
# Sequence length: 1024
# Labels: tensor([0, 1, 0, 0, 1, ...])
```

---

## Data Preprocessing

### Reading BAM Files

ChimeraLM only processes reads with SA (Supplementary Alignment) tags, which indicate potential chimeric reads.

```python
import pysam

# Open BAM file
bam = pysam.AlignmentFile("input.bam", "rb")

# Filter reads with SA tags
chimeric_candidates = []
for read in bam:
    if read.has_tag("SA"):
        chimeric_candidates.append(read)

print(f"Found {len(chimeric_candidates)} chimeric candidates")
```

### Sequence Extraction

```python
# Extract sequence from read
read = next(bam)
sequence = read.query_sequence  # DNA sequence string

# Tokenize
tokenizer = DNATokenizer()
token_ids = tokenizer.encode(sequence)

print(f"Sequence: {sequence[:50]}...")
print(f"Tokens: {token_ids[:50]}...")
```

---

## Performance Optimization

### DataLoader Settings

```python
# For CPU training
data_module = BamDataModule(
    train_data_path="train.bam",
    batch_size=32,
    num_workers=8,  # High worker count
    pin_memory=False
)

# For GPU training
data_module = BamDataModule(
    train_data_path="train.bam",
    batch_size=64,
    num_workers=4,  # Moderate worker count
    pin_memory=True  # Enable pin memory for faster GPU transfer
)
```

### Caching

```python
# Use HuggingFace datasets cache for faster repeated access
from datasets import load_from_disk

# Save preprocessed dataset
dataset = data_module.train_dataset
dataset.save_to_disk("cache/train_dataset")

# Load from cache
cached_dataset = load_from_disk("cache/train_dataset")
```

---

## See Also

- [Models API Reference](models.md) - Model architecture and training
- [CLI Commands](cli.md) - Command-line interface
- [Data Pipeline Architecture](../architecture/data-pipeline.md) - Data processing internals
- [Fine-Tuning Tutorial](../tutorials/fine-tuning.md) - Training guide
