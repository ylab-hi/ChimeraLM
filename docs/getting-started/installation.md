# Installation

Get ChimeraLM installed on your system in just a few minutes.

## Prerequisites

!!! info "Requirements"
\- **Python**: 3.10 or 3.11 (3.12 not yet supported)
\- **Operating System**: Linux, macOS, or Windows
\- **Optional**: CUDA-compatible GPU for accelerated inference

## Installation Methods

Choose your preferred installation method:

=== "pip"

````
The easiest way to install ChimeraLM:

```bash
pip install chimeralm
```

Verify the installation:

```bash
chimeralm --version
```
````

=== "conda"

````
Install using conda:

```bash
conda install -c conda-forge chimeralm
```

Or create a new environment:

```bash
conda create -n chimeralm python=3.10
conda activate chimeralm
pip install chimeralm
```
````

=== "uv"

````
Using the fast `uv` package manager:

```bash
uv pip install chimeralm
```

Or with a virtual environment:

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install chimeralm
```
````

=== "from source"

````
For development or the latest features:

```bash
# Clone the repository
git clone https://github.com/ylab-hi/ChimeraLM.git
cd ChimeraLM

# Install in development mode with uv
uv sync

# Verify installation
uv run chimeralm --version
```
````

## GPU Setup

!!! tip "GPU Acceleration"
For significantly faster predictions, install GPU support:

### CUDA (NVIDIA GPUs)

ChimeraLM automatically uses CUDA if available. Verify your GPU is detected:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

If CUDA is not detected, install PyTorch with CUDA support:

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

### MPS (Apple Silicon)

For M1/M2/M3 Macs, MPS acceleration is automatically enabled:

```bash
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

!!! warning "Python Version Compatibility"
ChimeraLM requires Python 3.10 or 3.11. Python 3.12 is not yet supported due to dependency constraints.

## Verification

Confirm ChimeraLM is installed correctly:

```bash
# Check version
chimeralm --version

# View available commands
chimeralm --help

# Test import (Python)
python -c "import chimeralm; print('ChimeraLM imported successfully')"
```

Expected output:

```text
ChimeraLM imported successfully
```

## Troubleshooting Installation

### Common Issues

??? question "ImportError: No module named 'chimeralm'"

````
**Solution**: Ensure you've activated the correct Python environment:

```bash
# Check which Python is being used
which python

# Reinstall in the current environment
pip install --force-reinstall chimeralm
```
````

??? question "CUDA out of memory during installation"

```
**Solution**: This is normal if testing GPU during install. Reduce batch size when running predictions.
```

??? question "PyTorch version conflict"

````
**Solution**: ChimeraLM requires PyTorch 2.5.1. Uninstall conflicting versions:

```bash
pip uninstall torch torchvision torchaudio
pip install chimeralm
```
````

??? question "Permission denied errors"

````
**Solution**: Install in user space without sudo:

```bash
pip install --user chimeralm
```
````

For more issues, see the [Troubleshooting Guide](troubleshooting.md).

## Next Steps

Now that ChimeraLM is installed, try the [Quick Start](quick-start.md) tutorial to run your first prediction!
