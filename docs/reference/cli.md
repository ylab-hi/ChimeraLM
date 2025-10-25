# CLI Commands Reference

Complete reference for all ChimeraLM command-line interface commands.

## Overview

ChimeraLM provides a Typer-based CLI with three main commands:

- **`predict`**: Predict chimeric reads in BAM files
- **`filter`**: Filter BAM files based on predictions
- **`web`**: Launch the web interface for visualization

## Command Structure

```bash
chimeralm [OPTIONS] COMMAND [ARGS]...
```

## Global Options

### `--version`, `-V`

Display ChimeraLM version information.

```bash
chimeralm --version
```

**Output:**
```text
ChimeraLM v0.1.0
```

### `--help`, `-h`

Display help information for all commands.

```bash
chimeralm --help
```

---

## `predict` Command

Predict whether reads in a BAM file are chimeric (label 1) or biological (label 0).

### Syntax

```bash
chimeralm predict [OPTIONS] DATA_PATH
```

### Arguments

#### `DATA_PATH`

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

#### `--gpus`, `-g`

**Type:** Integer
**Default:** `0` (CPU mode)

Number of GPUs to use for inference.

- `0`: CPU mode
- `1`: Single GPU (CUDA or MPS)
- `>1`: Multiple GPUs (if supported)

**Examples:**
```bash
# CPU mode (default)
chimeralm predict input.bam

# Use single GPU
chimeralm predict input.bam --gpus 1
chimeralm predict input.bam -g 1
```

#### `--output`, `-o`

**Type:** Path
**Default:** `{DATA_PATH}.predictions/`

Output directory for predictions.

**Examples:**
```bash
# Default output location
chimeralm predict input.bam
# Creates: input.bam.predictions/predictions.txt

# Custom output directory
chimeralm predict input.bam --output results/
chimeralm predict input.bam -o predictions/
# Creates: results/predictions.txt or predictions/predictions.txt
```

#### `--batch-size`, `-b`

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
chimeralm predict input.bam -b 8

# Large batch for high-memory GPU
chimeralm predict input.bam -g 1 -b 48
```

#### `--workers`, `-w`

**Type:** Integer
**Default:** `0` (main thread only)

Number of worker processes for data loading.

**Recommendations:**
- CPU mode: 4-8 workers
- GPU mode: 2-4 workers
- Default: 0 (single-threaded)

**Examples:**
```bash
# Single-threaded (default)
chimeralm predict input.bam

# Multi-threaded CPU
chimeralm predict input.bam -g 0 -w 8

# GPU with parallel data loading
chimeralm predict input.bam -g 1 -w 4
```

#### `--random`, `-r`

**Type:** Boolean flag
**Default:** `False` (deterministic)

Make predictions non-deterministic. By default, predictions are deterministic for reproducibility.

**Examples:**
```bash
# Deterministic predictions (default)
chimeralm predict input.bam

# Non-deterministic predictions
chimeralm predict input.bam --random
chimeralm predict input.bam -r
```

#### `--verbose`, `-v`

**Type:** Boolean flag
**Default:** `False`

Enable verbose logging for debugging.

**Examples:**
```bash
# Normal logging
chimeralm predict input.bam

# Verbose mode
chimeralm predict input.bam --verbose
chimeralm predict input.bam -v
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

**Short form:**
```bash
chimeralm predict input.bam -g 1 -b 24 -w 4 -o predictions/ -v
```

---

## `filter` Command

Filter BAM file to remove chimeric reads based on ChimeraLM predictions.

### Syntax

```bash
chimeralm filter [OPTIONS] BAM_PATH PREDICTIONS_PATH
```

### Arguments

#### `BAM_PATH`

**Type:** Path (required)

Path to input BAM file (same as used for prediction).

#### `PREDICTIONS_PATH`

**Type:** Path (required)

Path to predictions directory or predictions file.

**Examples:**
```bash
# Using predictions directory
chimeralm filter input.bam input.bam.predictions/

# Using predictions file directly
chimeralm filter input.bam input.bam.predictions/predictions.txt
```

### Options

#### `--output-prediction`, `-p`

**Type:** Boolean flag
**Default:** `False`

Write a consolidated predictions.txt file to the predictions directory. This merges all .txt files in the folder into a single file.

**Note:** The filtered BAM file is **always created** regardless of this flag. This flag only controls whether to write the consolidated predictions.txt.

**Examples:**
```bash
# Filter BAM (creates filtered.bam automatically)
chimeralm filter input.bam predictions/

# Filter BAM AND write consolidated predictions.txt
chimeralm filter input.bam predictions/ --output-prediction
chimeralm filter input.bam predictions/ -p
```

!!! info "Output Files Created"
    The filter command always creates:
    - `{BAM_PATH}.filtered.bam` - Filtered reads (unsorted)
    - `{BAM_PATH}.filtered.sorted.bam` - Sorted filtered BAM
    - `{BAM_PATH}.filtered.sorted.bam.bai` - BAM index

    Example: `input.bam` → `input.filtered.sorted.bam`

#### `--verbose`, `-v`

**Type:** Boolean flag
**Default:** `False`

Enable verbose logging.

**Examples:**
```bash
# Normal logging
chimeralm filter input.bam predictions/ -p

# Verbose mode
chimeralm filter input.bam predictions/ -p -v
```

### Output

**Always created:**
- `{BAM_PATH}.filtered.bam` - Unsorted filtered BAM
- `{BAM_PATH}.filtered.sorted.bam` - Sorted filtered BAM (final output)
- `{BAM_PATH}.filtered.sorted.bam.bai` - BAM index
- Console output with summary statistics

**With `--output-prediction` flag:**
- Additionally creates: `predictions.txt` in the predictions directory
- Consolidates all .txt files into a single predictions file

### Complete Example

```bash
# Step 1: Predict
chimeralm predict input.bam -g 1

# Step 2: Filter (creates sorted BAM automatically)
chimeralm filter input.bam input.bam.predictions/

# Step 3: Verify output
samtools view -c input.filtered.sorted.bam
```

---

## `web` Command

Launch the ChimeraLM web interface for interactive visualization and analysis.

### Syntax

```bash
chimeralm web [OPTIONS]
```

### Options

Currently, the `web` command has no additional options besides `--help`.

### Usage

```bash
# Launch web interface
chimeralm web
```

**Expected behavior:**
- Starts a local web server
- Opens browser to the ChimeraLM web interface
- Provides interactive tools for prediction and visualization

**Typical output:**
```text
Starting ChimeraLM web interface...
Server running at: http://localhost:8000
Open your browser to visualize predictions
Press Ctrl+C to stop
```

### Web Interface Features

The web interface typically provides:

- **Interactive prediction**: Upload BAM files for prediction
- **Visualization**: View prediction results graphically
- **Batch processing**: Process multiple files
- **Export results**: Download predictions and filtered BAM files

!!! tip "Web Interface Access"
    If the browser doesn't open automatically, manually navigate to `http://localhost:8000` (or the URL shown in the terminal).

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
CUDA_VISIBLE_DEVICES=0 chimeralm predict input.bam -g 1

# Use GPU 1
CUDA_VISIBLE_DEVICES=1 chimeralm predict input.bam -g 1

# Use multiple GPUs
CUDA_VISIBLE_DEVICES=0,1 chimeralm predict input.bam -g 2
```

---

## Common Workflows

### Basic Prediction and Filtering

```bash
# 1. Predict
chimeralm predict input.bam -g 1 -b 24

# 2. Filter (automatically creates sorted BAM)
chimeralm filter input.bam input.bam.predictions/

# 3. Verify
samtools view -c input.filtered.sorted.bam
```

### Batch Processing

```bash
# Process multiple files
for bam in data/*.bam; do
    echo "Processing $bam..."
    chimeralm predict $bam -g 1 -b 24
    chimeralm filter $bam ${bam}.predictions/
    # Output: ${bam%.bam}.filtered.sorted.bam
done
```

### Using Web Interface

```bash
# Launch web interface for interactive analysis
chimeralm web

# Access at http://localhost:8000
# Upload BAM files through the web UI
```

### High-Performance Prediction

```bash
# Maximize GPU utilization
chimeralm predict input.bam \
    --gpus 1 \
    --batch-size 64 \
    --workers 4 \
    --output predictions/

# Then filter (creates sorted BAM)
chimeralm filter input.bam predictions/
```

---

## Shell Completion

Install shell completion for easier command usage:

```bash
# Install completion
chimeralm --install-completion

# Show completion script
chimeralm --show-completion
```

Supports: Bash, Zsh, Fish, PowerShell

---

## See Also

- [Quick Start Tutorial](../getting-started/quick-start.md)
- [Performance Optimization](../tutorials/performance-optimization.md)
- [BAM Filtering Tutorial](../tutorials/bam-filtering.md)
- [Models API Reference](models.md)
