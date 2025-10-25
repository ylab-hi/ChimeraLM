# CLI Commands Reference

Complete reference for all ChimeraLM command-line interface commands.

## Overview

ChimeraLM provides a Typer-based CLI with three main commands:

- **`predict`**: Predict chimeric reads in BAM files
- **`filter`**: Filter BAM files based on predictions
- **`finetune`**: Fine-tune the model on custom data

## Command Structure

```bash
chimeralm [OPTIONS] COMMAND [ARGS]...
```

## Global Options

### `--version`

Display ChimeraLM version information.

```bash
chimeralm --version
```

**Output:**
```text
ChimeraLM v0.1.0
```

### `--help`

Display help information for all commands.

```bash
chimeralm --help
```

---

## `predict` Command

Predict whether reads in a BAM file are chimeric (label 1) or biological (label 0).

### Syntax

```bash
chimeralm predict [OPTIONS] BAM_FILE
```

### Arguments

#### `BAM_FILE`

**Type:** Path (required)

Path to input BAM file. File must:
- Be a valid BAM/SAM file
- Contain reads with SA tags (supplementary alignment tags)
- Be readable by pysam

**Example:**
```bash
chimeralm predict input.bam
chimeralm predict /path/to/data/sample.bam
```

### Options

#### `--ckpt PATH`

**Type:** Path
**Default:** `yangliz5/chimeralm` (Hugging Face Hub)

Path to model checkpoint file or Hugging Face model ID.

**Examples:**
```bash
# Use default pretrained model
chimeralm predict input.bam

# Use local checkpoint
chimeralm predict input.bam --ckpt /path/to/checkpoint.ckpt

# Use specific Hugging Face model
chimeralm predict input.bam --ckpt username/model-name
```

#### `--gpus INTEGER`

**Type:** Integer
**Default:** Auto-detect (1 if GPU available, 0 otherwise)

Number of GPUs to use for inference.

- `0`: CPU mode
- `1`: Single GPU (CUDA or MPS)
- `>1`: Currently not supported (use `1`)

**Examples:**
```bash
# Auto-detect (recommended)
chimeralm predict input.bam

# Force CPU mode
chimeralm predict input.bam --gpus 0

# Use GPU
chimeralm predict input.bam --gpus 1
```

#### `--batch-size INTEGER`

**Type:** Integer
**Default:** `12`

Number of reads to process in each batch. Larger batches improve GPU utilization but require more memory.

**Recommendations:**
- CPU: 12-32
- GPU (8GB): 12-16
- GPU (16GB): 24-32
- GPU (24GB+): 48-64

**Examples:**
```bash
# Default batch size
chimeralm predict input.bam

# Small batch for limited memory
chimeralm predict input.bam --batch-size 8

# Large batch for high-memory GPU
chimeralm predict input.bam --gpus 1 --batch-size 48
```

#### `--workers INTEGER`

**Type:** Integer
**Default:** `0` (main thread only)

Number of worker processes for data loading.

**Recommendations:**
- CPU mode: 4-8 workers
- GPU mode: 2-4 workers

**Examples:**
```bash
# Single-threaded (default)
chimeralm predict input.bam

# Multi-threaded CPU
chimeralm predict input.bam --gpus 0 --workers 8

# GPU with parallel data loading
chimeralm predict input.bam --gpus 1 --workers 4
```

#### `--max-sample INTEGER`

**Type:** Integer
**Default:** `None` (process all reads)

Maximum number of reads to process. Useful for testing or processing large files in chunks.

**Examples:**
```bash
# Process all reads
chimeralm predict input.bam

# Process first 1000 reads
chimeralm predict input.bam --max-sample 1000

# Process first 100K reads for testing
chimeralm predict input.bam --max-sample 100000
```

#### `--output PATH`

**Type:** Path
**Default:** `{BAM_FILE}.predictions/`

Output directory for predictions.

**Examples:**
```bash
# Default output location
chimeralm predict input.bam
# Creates: input.bam.predictions/predictions.txt

# Custom output directory
chimeralm predict input.bam --output results/predictions/
# Creates: results/predictions/predictions.txt
```

#### `--verbose / --no-verbose`

**Type:** Boolean
**Default:** `False`

Enable verbose logging for debugging.

**Examples:**
```bash
# Normal logging
chimeralm predict input.bam

# Verbose mode
chimeralm predict input.bam --verbose
```

### Output Format

Predictions are saved as a tab-separated text file:

```text
read_name<TAB>label
```

**Example:**
```text
m54329U_200919_012139/4194729/ccs	0
m54329U_200919_012139/4194826/ccs	1
m54329U_200919_012139/4194958/ccs	0
```

**Labels:**
- `0`: Biological read (keep for analysis)
- `1`: Chimeric artifact (remove from analysis)

### Complete Example

```bash
chimeralm predict input.bam \
    --gpus 1 \
    --batch-size 24 \
    --workers 4 \
    --output predictions/ \
    --verbose
```

---

## `filter` Command

Filter BAM file to remove chimeric reads based on ChimeraLM predictions.

### Syntax

```bash
chimeralm filter [OPTIONS] BAM_FILE PREDICTIONS_DIR
```

### Arguments

#### `BAM_FILE`

**Type:** Path (required)

Path to input BAM file (same as used for prediction).

#### `PREDICTIONS_DIR`

**Type:** Path (required)

Path to predictions directory containing `predictions.txt`.

**Example:**
```bash
chimeralm filter input.bam input.bam.predictions/
```

### Options

#### `--output-prediction PATH`

**Type:** Path
**Default:** `{BAM_FILE}.filtered.bam`

Path to output filtered BAM file.

**Examples:**
```bash
# Default output
chimeralm filter input.bam predictions/
# Creates: input.bam.filtered.bam

# Custom output path
chimeralm filter input.bam predictions/ --output-prediction clean.bam
# Creates: clean.bam
```

### Output

Creates three files:

1. **Filtered BAM**: Contains only biological reads (label 0)
2. **BAM index**: `.bam.bai` file for indexed access
3. **Sorted BAM**: Output is sorted and indexed

### Complete Example

```bash
# Predict
chimeralm predict input.bam --gpus 1

# Filter
chimeralm filter input.bam input.bam.predictions/ \
    --output-prediction clean.bam

# Verify
samtools view -c clean.bam
```

---

## `finetune` Command

Fine-tune ChimeraLM on custom labeled data.

### Syntax

```bash
chimeralm finetune [OPTIONS]
```

### Options

#### `--train-data PATH`

**Type:** Path (required)

Path to training BAM file with labeled reads.

**Example:**
```bash
chimeralm finetune --train-data labeled_data.bam
```

#### `--val-data PATH`

**Type:** Path
**Default:** `None` (auto-split from train-data)

Path to validation BAM file.

**Example:**
```bash
chimeralm finetune --train-data train.bam --val-data val.bam
```

#### `--test-data PATH`

**Type:** Path
**Default:** `None` (auto-split from train-data)

Path to test BAM file.

**Example:**
```bash
chimeralm finetune \
    --train-data train.bam \
    --val-data val.bam \
    --test-data test.bam
```

#### `--epochs INTEGER`

**Type:** Integer
**Default:** `50`

Number of training epochs.

**Example:**
```bash
chimeralm finetune --train-data data.bam --epochs 100
```

#### `--batch-size INTEGER`

**Type:** Integer
**Default:** `12`

Training batch size.

**Example:**
```bash
chimeralm finetune --train-data data.bam --batch-size 32
```

#### `--gpus INTEGER`

**Type:** Integer
**Default:** Auto-detect

Number of GPUs for training.

**Example:**
```bash
chimeralm finetune --train-data data.bam --gpus 1
```

#### `--seed INTEGER`

**Type:** Integer
**Default:** `None` (random)

Random seed for reproducibility.

**Example:**
```bash
chimeralm finetune --train-data data.bam --seed 42
```

#### `--no-test`

**Type:** Boolean flag
**Default:** `False` (test is run)

Skip final test evaluation.

**Example:**
```bash
chimeralm finetune --train-data data.bam --no-test
```

#### `--model STRING`

**Type:** String
**Default:** `chimeralm`

Model architecture (hidden option for advanced users).

**Choices:** `chimeralm`, `cnn`, `hyena`, `mamba`

**Example:**
```bash
chimeralm finetune --train-data data.bam --model cnn
```

#### `-r, --override KEY=VALUE`

**Type:** String (repeatable)

Hydra configuration overrides for advanced customization.

**Examples:**
```bash
# Single override
chimeralm finetune --train-data data.bam -r model.optimizer.lr=0.0001

# Multiple overrides
chimeralm finetune --train-data data.bam \
    -r model.optimizer.lr=0.0001 \
    -r trainer.precision=16-mixed \
    -r data.num_workers=8
```

**Common Overrides:**
- `model.optimizer.lr=FLOAT`: Learning rate
- `model.optimizer.weight_decay=FLOAT`: Weight decay
- `trainer.precision=STRING`: Precision mode (`16-mixed`, `32`)
- `data.num_workers=INT`: Data loading workers
- `data.train_val_test_split=LIST`: Split ratios (e.g., `[0.8,0.1,0.1]`)

### Complete Example

```bash
chimeralm finetune \
    --train-data labeled_data.bam \
    --epochs 100 \
    --batch-size 32 \
    --gpus 1 \
    --seed 42 \
    -r model.optimizer.lr=0.0001 \
    -r trainer.precision=16-mixed
```

### Output

Training checkpoints and logs are saved to:
```
logs/train/runs/YYYY-MM-DD_HH-MM-SS/
├── checkpoints/
│   ├── best.ckpt          # Best model by validation loss
│   └── last.ckpt          # Latest checkpoint
├── config.yaml            # Full configuration
└── tensorboard/           # TensorBoard logs
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error (file not found, invalid input, etc.) |
| 2 | CUDA out of memory (reduce `--batch-size`) |
| 130 | User interrupt (Ctrl+C) |

---

## Environment Variables

### `CUDA_VISIBLE_DEVICES`

Control which GPUs are visible to ChimeraLM.

```bash
# Use GPU 0
CUDA_VISIBLE_DEVICES=0 chimeralm predict input.bam

# Use GPU 1
CUDA_VISIBLE_DEVICES=1 chimeralm predict input.bam

# Use multiple GPUs (not fully supported yet)
CUDA_VISIBLE_DEVICES=0,1 chimeralm predict input.bam
```

### `WANDB_API_KEY`

Weights & Biases API key for experiment tracking during fine-tuning.

```bash
export WANDB_API_KEY="your_key_here"
chimeralm finetune --train-data data.bam
```

### `HF_HOME`

Hugging Face cache directory for downloaded models.

```bash
export HF_HOME="/path/to/cache"
chimeralm predict input.bam
```

---

## Common Workflows

### Basic Prediction

```bash
# 1. Predict
chimeralm predict input.bam --gpus 1

# 2. Filter
chimeralm filter input.bam input.bam.predictions/ \
    --output-prediction filtered.bam

# 3. Verify
samtools view -c filtered.bam
```

### Batch Processing

```bash
# Process multiple files
for bam in data/*.bam; do
    echo "Processing $bam..."
    chimeralm predict $bam --gpus 1 --batch-size 24
    chimeralm filter $bam ${bam}.predictions/ \
        --output-prediction filtered_$(basename $bam)
done
```

### Fine-Tuning Workflow

```bash
# 1. Fine-tune
chimeralm finetune --train-data labeled.bam --epochs 100 --gpus 1

# 2. Find checkpoint
CKPT=$(ls -t logs/train/runs/*/checkpoints/best.ckpt | head -1)

# 3. Predict with fine-tuned model
chimeralm predict input.bam --ckpt $CKPT --gpus 1
```

---

## See Also

- [Quick Start Tutorial](../getting-started/quick-start.md)
- [Fine-Tuning Tutorial](../tutorials/fine-tuning.md)
- [Performance Optimization](../tutorials/performance-optimization.md)
- [Models API Reference](models.md)
