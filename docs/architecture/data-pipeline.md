# Data Pipeline

ChimeraLM data processing pipeline and tokenization strategy.

## Pipeline Overview

```
BAM File
  ↓
Read with pysam
  ↓
Filter SA tags (chimeric candidates)
  ↓
Extract DNA sequences
  ↓
Tokenize (A→1, C→2, G→3, T→4, N→0)
  ↓
Batch and pad
  ↓
Create DataLoader
  ↓
Model training/inference
```

## BAM File Processing

### Reading BAM Files

ChimeraLM uses pysam to read BAM/SAM files:

```python
import pysam

# Open BAM file
bam = pysam.AlignmentFile("input.bam", "rb")

# Iterate through reads
for read in bam:
    if read.has_tag("SA"):  # Supplementary alignment tag
        # Process chimeric candidate
        process_read(read)
```

### SA Tag Filtering

**Why SA tags?**

- **SA (Supplementary Alignment)**: Indicates chimeric or split alignment
- Reads with SA tags are chimeric candidates
- Biological reads rarely have SA tags

### Sequence Extraction

```python
# Get DNA sequence from read
sequence = read.query_sequence  # "ACGTACGT..."
quality = read.query_qualities   # Phred scores
```

## Tokenization

### Character-Level Tokenization

| Character | Token ID |
|-----------|----------|
| N | 0 |
| A | 1 |
| C | 2 |
| G | 3 |
| T | 4 |
| [PAD] | 5 |

**Example:**
```python
sequence = "ACGTNNACGT"
tokens = [1, 2, 3, 4, 0, 0, 1, 2, 3, 4]
```

### Vocabulary

- **Size**: 8 tokens (A, C, G, T, N, PAD, CLS, SEP)
- **Special tokens**: PAD (5), CLS (6), SEP (7)
- **Unknown bases**: Mapped to N (0)

## Batching and Padding

### Padding Strategy

```python
# Sequences of varying length
seq1 = [1, 2, 3, 4]         # Length 4
seq2 = [3, 3, 2, 2, 1, 1]   # Length 6

# Pad to max_length=8
seq1_padded = [1, 2, 3, 4, 5, 5, 5, 5]
seq2_padded = [3, 3, 2, 2, 1, 1, 5, 5]
```

### Batch Structure

```python
batch = {
    "input_ids": torch.Tensor,        # [batch_size, max_length]
    "labels": torch.Tensor,           # [batch_size]
    "attention_mask": torch.Tensor   # [batch_size, max_length]
}
```

## Data Loading

### DataModule

```python
from chimeralm.data.bam import BamDataModule

data_module = BamDataModule(
    train_data_path="train.bam",
    batch_size=32,
    num_workers=4,
    max_length=1024
)

# Setup creates dataloaders
data_module.setup("fit")

# Get loaders
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()
```

### Data Splitting

**Auto-split** (default):
- Train: 70%
- Validation: 20%
- Test: 10%

**Manual split**:
- Provide separate BAM files for train/val/test

## Performance Optimization

### Multi-Worker Data Loading

```python
# CPU: 4-8 workers
data_module = BamDataModule(
    train_data_path="train.bam",
    num_workers=8
)

# GPU: 2-4 workers
data_module = BamDataModule(
    train_data_path="train.bam",
    num_workers=4
)
```

### Caching

HuggingFace Datasets automatically caches processed data for faster repeated access.

## See Also

- [Data API Reference](../reference/data.md)
- [Architecture Overview](overview.md)
- [BAM Filtering Tutorial](../tutorials/bam-filtering.md)
