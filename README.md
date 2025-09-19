# <img src="./docs/logo.png" alt="logo" height="100"/> **ChimeraLM** [![social](https://img.shields.io/github/stars/ylab-hi/ChimeraLM?style=social)](https://github.com/ylab-hi/ChimeraLM/stargazers)

# ChimeraLM

A genomic language model to identify chimera artifacts introduced by whole genome amplification (WGA).

## Overview

ChimeraLM is a deep learning model designed to detect artificial chimeric reads that arise during whole genome amplification processes.

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

## CLI Usage

ChimeraLM provides a Python CLI with two main commands for chimeric read detection and filtering.

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

### Performance Tips

1. **GPU Usage**: Use `--gpus 1` for faster processing if CUDA is available
2. **Batch Size**: Increase `--batch-size` for better GPU utilization (e.g., 24-32)
3. **Memory**: Monitor memory usage with large batch sizes
4. **Threading**: Adjust `--workers` based on your system's CPU cores

### Output Files

The `predict` command generates:

- Prediction results in the specified output directory
- Filtered and sorted BAM file with index (automatically created)

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
