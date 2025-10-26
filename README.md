<div align="center">

<img src="./docs/logo-pixel.svg" alt="ChimeraLM Logo" width="128" height="128"/>

# ChimeraLM

**A Genomic Language Model for Detecting WGA Chimeric Artifacts**

[![python](https://img.shields.io/badge/Python-3776AB.svg?style=for-the-badge&logo=Python&logoColor=white)](https://www.python.org/)
[![pypi](https://img.shields.io/pypi/v/chimeralm.svg?style=for-the-badge)][pypi]
[![pyversion](https://img.shields.io/pypi/pyversions/chimeralm?style=for-the-badge)][pypi]
[![download](https://img.shields.io/pypi/dm/chimeralm?logo=pypi&label=downloads&style=for-the-badge)][pypi]
[![ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg?style=for-the-badge)](https://github.com/charliermarsh/ruff)

[![release](https://img.shields.io/github/release-date/ylab-hi/chimeralm?style=for-the-badge)](https://github.com/ylab-hi/chimeralm/releases)
[![stars](https://img.shields.io/github/stars/ylab-hi/ChimeraLM?style=for-the-badge&logo=github)](https://github.com/ylab-hi/ChimeraLM/stargazers)
[![activity](https://img.shields.io/github/commit-activity/m/ylab-hi/chimeralm?style=for-the-badge)][repo]
[![lastcommit](https://img.shields.io/github/last-commit/ylab-hi/chimeralm?style=for-the-badge)][repo]

[Installation](#installation) • [Quick Start](#quick-start) • [Documentation](#cli-usage) • [Citation](#citation)

</div>

______________________________________________________________________

A genomic language model to identify chimera artifacts introduced by whole genome amplification (WGA).

## Overview

ChimeraLM is a deep learning-powered genomic language model that detects artificial chimeric reads arising from whole genome amplification (WGA) processes. Built with PyTorch Lightning and optimized for modern GPUs, it provides fast and accurate identification of chimeric artifacts in BAM files.

### Key Features

- **High Accuracy**: Deep learning model trained on real WGA data
- **GPU Accelerated**: Optimized for CUDA, MPS (Apple Silicon), and CPU
- **Easy to Use**: Simple CLI with sensible defaults
- **Fast Processing**: Batch inference with configurable parallelism
- **Web Interface**: Interactive web UI for visualization and analysis
- **Production Ready**: Includes filtering, sorting, and indexing of BAM files

## Installation

### Install from PyPI

```bash
pip install chimeralm
```

### Install from Source

```bash
# Clone the repository
git clone https://github.com/ylab-hi/ChimeraLM.git
cd ChimeraLM

# Install in development mode with uv
uv sync

uv run chimeralm --version
```

## Quick Start

Detect chimeric reads in your BAM file:

```bash
# Install ChimeraLM
pip install chimeralm

# Predict chimeric reads (CPU)
chimeralm predict your_data.bam

# Predict with GPU acceleration
chimeralm predict your_data.bam --gpus 1 --batch-size 24
```

## CLI Usage

ChimeraLM provides a Python CLI with three main commands:

- `predict`: Detect chimeric reads using pre-trained models
- `filter`: Filter BAM files based on predictions
- `web`: Launch interactive web interface

### Command Structure

```bash
chimeralm [OPTIONS] COMMAND [ARGS]...
```

### Available Commands

#### `predict` - Detect Chimeric Reads

Predict chimeric reads in a BAM file using the pre-trained ChimeraLM model.

```bash
chimeralm predict [OPTIONS] DATA_PATH
```

**Arguments:**

- `DATA_PATH`: Path to the input BAM file

**Options:**

- `-g, --gpus INTEGER`: Number of GPUs to use (default: 0)
- `-o, --output PATH`: Output path for predictions (default: `{input}.predictions`)
- `-b, --batch-size INTEGER`: Batch size for processing (default: 12)
- `-w, --workers INTEGER`: Number of worker threads (default: 0)
- `-v, --verbose`: Enable verbose output
- `-m, --max-sample INTEGER`: Maximum number of samples to process
- `-l, --limit-batches INTEGER`: Limit prediction batches
- `-p, --progress-bar`: Show progress bar
- `--random-seed`: Make prediction non-deterministic

**Examples:**

```bash
# Basic prediction on CPU
chimeralm predict input.bam

# Prediction with GPU acceleration
chimeralm predict input.bam --gpus 1 --batch-size 24

# Prediction with custom output path and progress bar
chimeralm predict input.bam --output results/ --progress-bar --verbose
```

#### `filter` - Filter BAM Files

Filter BAM files based on prediction results.

```bash
chimeralm filter [OPTIONS] INPUT_BAM PREDICTIONS_DIR
```

**Arguments:**

- `INPUT_BAM`: Path to the input BAM file
- `PREDICTIONS_DIR`: Directory containing prediction results

**Options:**

- `-o, --output-prediction PATH`: Output path for filtered BAM file

**Example:**

```bash
chimeralm filter input.bam predictions/ --output-prediction filtered.bam
```

#### `web` - Launch Web Interface

Launch an interactive web interface for visualizing and analyzing chimeric reads.

```bash
chimeralm web
```

This command starts a local web server that provides:

- Interactive visualization of predictions
- Analysis dashboards and metrics
- Easy-to-use interface for non-technical users

### Performance Tips

1. **GPU Usage**: Use `--gpus 1` for faster processing if CUDA is available
2. **Batch Size**: Increase `--batch-size` for better GPU utilization (e.g., 24-32)
3. **Memory**: Monitor memory usage with large batch sizes
4. **Threading**: Adjust `--workers` based on your system's CPU cores

### Output Files

**Prediction outputs:**

- `{output_dir}/predictions.txt`: Tab-separated file with read names and predicted labels (0=biological, 1=chimeric)
- `{input}.filtered.sorted.bam`: BAM file with chimeric reads removed (auto-generated by filter command)
- `{input}.filtered.sorted.bam.bai`: BAM index file

### Troubleshooting

**Common Issues:**

1. **CUDA out of memory**: Reduce `--batch-size` or use CPU mode
2. **Slow processing**: Enable GPU acceleration with `--gpus 1`
3. **Missing dependencies**: Run `uv sync` to install all dependencies

**Debug Mode:**
Use `--verbose` flag to get detailed logging information about the prediction process.

### Version Information

```bash
chimeralm --version
```

### Getting Help

```bash
# General help
chimeralm --help

# Command-specific help
chimeralm predict --help
```

## Citation

If you use ChimeraLM in your research, please cite:

```bibtex
@software{chimeralm2025,
  title={ChimeraLM: A genomic language model to identify chimera artifacts},
  author={Li, Yangyang, Guo, Qingxiang and Yang, Rendong},
  year={2025},
  url={https://github.com/ylab-hi/ChimeraLM}
}
```

## License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

[pypi]: https://pypi.org/project/chimeralm/
[repo]: https://github.com/ylab-hi/chimeralm
