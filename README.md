<div align="center">

<img src="./docs/logo-pixel.svg" alt="ChimeraLM Logo" width="128" height="128"/>

# ChimeraLM

**A Genomic Language Model for Detecting WGA Chimeric Artifacts**

[![python](https://img.shields.io/badge/Python-3776AB.svg?style=for-the-badge&logo=Python&logoColor=white)](https://www.python.org/)
[![pypi](https://img.shields.io/pypi/v/chimeralm.svg?style=for-the-badge)][pypi]
[![pyversion](https://img.shields.io/pypi/pyversions/chimeralm?style=for-the-badge)][pypi]
[![download](https://img.shields.io/pypi/dm/chimeralm?logo=pypi&label=downloads&style=for-the-badge)][pypi]
[![ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg?style=for-the-badge)](https://github.com/charliermarsh/ruff)

[![release](https://img.shields.io/github/release-date/ylab-hi/ChimeraLM?style=for-the-badge)](https://github.com/ylab-hi/ChimeraLM/releases)
[![stars](https://img.shields.io/github/stars/ylab-hi/ChimeraLM?style=for-the-badge&logo=github)](https://github.com/ylab-hi/ChimeraLM/stargazers)
[![activity](https://img.shields.io/github/commit-activity/m/ylab-hi/chimeralm?style=for-the-badge)][repo]
[![lastcommit](https://img.shields.io/github/last-commit/ylab-hi/chimeralm?style=for-the-badge)][repo]

[Installation](#installation) • [Quick Start](#quick-start) • [Documentation](https://ylab-hi.github.io/ChimeraLM/) • [Citation](#citation)

</div>

______________________________________________________________________

A deep learning-powered tool to identify chimeric artifacts introduced by whole genome amplification (WGA).

## Installation

```bash
pip install chimeralm
```

**Requirements:** Python 3.10, 3.11 and 3.12

For GPU support, installation instructions, and troubleshooting, see the [Installation Guide](https://ylab-hi.github.io/ChimeraLM/getting-started/installation/).

## Quick Start

```bash
# Predict chimeric reads (CPU)
chimeralm predict your_data.bam

# Predict with GPU acceleration
chimeralm predict your_data.bam --gpus 1 --batch-size 24

# Filter BAM to remove chimeric reads
chimeralm filter your_data.bam your_data.predictions
```

**Output:**

- Predictions: Tab-separated file with read names and labels (0=biological, 1=chimeric)
- Filtered BAM: `{input}.filtered.sorted.bam` with chimeric reads removed

**Need more help?** See the [Quick Start Tutorial](https://ylab-hi.github.io/ChimeraLM/getting-started/quick-start/) for a complete walkthrough.

## Documentation

Full documentation is available at **[ylab-hi.github.io/ChimeraLM](https://ylab-hi.github.io/ChimeraLM/)**

**Key Resources:**

- [Installation Guide](https://ylab-hi.github.io/ChimeraLM/getting-started/installation/) - Setup with pip, conda, uv, or from source
- [Quick Start Tutorial](https://ylab-hi.github.io/ChimeraLM/getting-started/quick-start/) - Your first prediction in 15 minutes
- [CLI Reference](https://ylab-hi.github.io/ChimeraLM/reference/cli/) - Complete command documentation
- [BAM Filtering Tutorial](https://ylab-hi.github.io/ChimeraLM/tutorials/bam-filtering/) - Comprehensive filtering guide
- [Performance Optimization](https://ylab-hi.github.io/ChimeraLM/tutorials/performance-optimization/) - Speed up your analysis
- [Troubleshooting](https://ylab-hi.github.io/ChimeraLM/getting-started/troubleshooting/) - Common issues and solutions

## Features

- **High Accuracy**: Deep learning model trained on real WGA data
- **GPU Accelerated**: Optimized for CUDA, MPS (Apple Silicon), and CPU
- **Easy to Use**: Simple CLI with sensible defaults
- **Fast Processing**: Batch inference with configurable parallelism
- **Web Interface**: Interactive web UI for visualization and analysis
- **Production Ready**: Includes filtering, sorting, and indexing of BAM files

## Contributing

Contributions are welcome! See our [Contributing Guide](https://ylab-hi.github.io/ChimeraLM/contributing/development-setup/) for development setup and guidelines.

## Citation

If you use ChimeraLM in your research, please cite:

```bibtex
@software{chimeralm2025,
  title={ChimeraLM: A genomic language model to identify chimera artifacts},
  author={Li, Yangyang and Guo, Qingxiang and Yang, Rendong},
  year={2025},
  url={https://github.com/ylab-hi/ChimeraLM}
}
```

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

[pypi]: https://pypi.org/project/chimeralm/
[repo]: https://github.com/ylab-hi/chimeralm
