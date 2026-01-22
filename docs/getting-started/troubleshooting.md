# Troubleshooting

Common issues and solutions for ChimeraLM users.

## Installation Issues

### Python Version Errors

??? question "ModuleNotFoundError or ImportError after installation"

   **Symptom**: `ModuleNotFoundError: No module named 'chimeralm'`

   **Cause**: Wrong Python environment or installation failed

   **Solution**:

   ```bash
   # Check Python version (must be 3.10 or 3.11)
   python --version

   # Verify pip is using correct Python
   which pip
   python -m pip --version

   # Reinstall in current environment
   python -m pip install --force-reinstall chimeralm
   ```

??? question "UnsupportedPython version: requires Python 3.10 or 3.11"

   **Symptom**: Installation fails with Python version error

   **Cause**: ChimeraLM doesn't support Python 3.12 yet

   **Solution**:

   ```bash
   # Create environment with Python 3.11
   conda create -n chimeralm python=3.11
   conda activate chimeralm
   pip install chimeralm
   ```

### Dependency Conflicts

??? question "ERROR: pip's dependency resolver does not currently take into account all the packages"

   **Symptom**: Pip reports dependency conflicts during installation

   **Solution**:

   ```bash
   # Install in a clean environment
   python -m venv chimeralm_env
   source chimeralm_env/bin/activate  # On Windows: chimeralm_env\Scripts\activate
   pip install chimeralm
   ```

## Runtime Issues

### CUDA / GPU Problems

??? question "CUDA out of memory"

    **Symptom**: `RuntimeError: CUDA out of memory`

    **Cause**: Batch size too large for your GPU

    **Solution**:
    ```bash
    # Reduce batch size (default is 12)
    chimeralm predict input.bam --gpus 1 --batch-size 8

    # Or use CPU mode
    chimeralm predict input.bam --gpus 0
    ```

??? question "GPU not detected despite having CUDA"

    **Symptom**: ChimeraLM runs on CPU even with `--gpus 1`

    **Cause**: PyTorch not installed with CUDA support

    **Solution**:
    ```bash
    # Check CUDA availability
    python -c "import torch; print(torch.cuda.is_available())"

    # If False, reinstall PyTorch with CUDA
    pip uninstall torch torchvision torchaudio
    pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
    pip install chimeralm
    ```

### Model Loading Issues

??? question "HTTPError: 404 Client Error when loading model"

    **Symptom**: `HTTPError: 404 Client Error: Not Found for url`

    **Cause**: Cannot connect to Hugging Face Hub to download model

    **Solution**:
    ```bash
    # Check internet connection
    curl -I https://huggingface.co

    # Use local model checkpoint
    chimeralm predict input.bam --ckpt /path/to/local/checkpoint.ckpt

    # Or download model manually first
    python -c "from transformers import AutoModel; AutoModel.from_pretrained('yangliz5/chimeralm')"
    ```

??? question "Model not found error"

    **Symptom**: `Model yangliz5/chimeralm not found`

    **Cause**: Model not cached locally and internet unavailable

    **Solution**: Download the model when you have internet, it will be cached:
    ```bash
    # Pre-download model
    python -c "from chimeralm.models.lm import ChimeraLM; ChimeraLM.from_pretrained('yangliz5/chimeralm')"
    ```

### BAM File Issues

??? question "ValueError: BAM file not readable or invalid format"

    **Symptom**: `ValueError: BAM file not readable`

    **Cause**: File is corrupted, not a valid BAM, or lacks SA tags

    **Solution**:
    ```bash
    # Verify BAM file is valid
    samtools view -H input.bam | head

    # Check for SA tags (chimeric indicator)
    samtools view input.bam | grep -c "SA:Z:"

    # If no SA tags found
    # ChimeraLM only processes reads with SA tags (chimeric candidates)
    # Make sure your WGA data includes supplementary alignments
    ```

??? question "PermissionError: [Errno 13] Permission denied"

    **Symptom**: Cannot read or write BAM files

    **Solution**:
    ```bash
    # Check file permissions
    ls -l input.bam

    # Add read permission
    chmod +r input.bam

    # Check write permission for output directory
    ls -ld predictions/
    chmod +w predictions/
    ```

## Performance Issues

??? question "Predictions are very slow (>1 minute for 1000 reads)"

    **Symptom**: Predictions take much longer than expected

    **Cause**: Running on CPU or batch size too small

    **Solution**:
    ```bash
    # Enable GPU if available
    chimeralm predict input.bam --gpus 1 --batch-size 24

    # Increase worker threads (if using CPU)
    chimeralm predict input.bam --gpus 0 --workers 4
    ```

??? question "High memory usage"

    **Symptom**: System runs out of RAM during prediction

    **Solution**:
    ```bash
    # Reduce batch size
    chimeralm predict input.bam --batch-size 4

    # Process in smaller chunks with --max-sample
    chimeralm predict input.bam --max-sample 500
    ```

## Output Issues

??? question "Predictions file is empty"

    **Symptom**: predictions.txt created but has no content

    **Cause**: No reads with SA tags found in BAM file

    **Solution**:
    ```bash
    # Verify input BAM has chimeric candidates
    samtools view input.bam | grep "SA:Z:" | wc -l

    # If count is 0, your BAM has no chimeric candidates
    # This is expected for non-WGA data
    ```

??? question "FilteredBAM has same size as input"

    **Symptom**: `chimeralm filter` produces output same size as input

    **Cause**: No chimeric reads detected (all labeled as 0)

    **Check**:
    ```bash
    # Count chimeric predictions
    grep -c "1$" predictions/predictions.txt

    # If count is 0, no chimeric reads detected
    # This could be normal for high-quality data
    ```

## Working with RNA Sequencing Data?

!!! note "ChimeraLM is for WGA DNA Sequencing"

    ChimeraLM is specifically designed for detecting chimeric artifacts from **whole genome amplification (WGA)** in DNA sequencing data.
    
    **For RNA sequencing data**, you should use [**DeepChopper**](https://ylab-hi.github.io/DeepChopper/), a specialized tool designed to identify chimera artifacts caused by internal adapter sequences in Nanopore direct RNA sequencing (dRNA-seq).
    
    **Key Differences:**
    
    - **ChimeraLM**: DNA sequencing, WGA-induced chimeras
    - **DeepChopper**: RNA sequencing, adapter-induced chimeras

## General Help

### Enable Verbose Logging

For debugging, enable detailed output:

```bash
chimeralm predict input.bam --verbose
```

### Check ChimeraLM Version

Ensure you're using the latest version:

```bash
# Check current version
chimeralm --version

# Update to latest
pip install --upgrade chimeralm
```

### System Information

Collect system info for bug reports:

```bash
# Python version
python --version

# PyTorch version and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# ChimeraLM version
chimeralm --version
```

## Getting Further Help

If your issue isn't covered here:

1. **Check existing issues**: [GitHub Issues](https://github.com/ylab-hi/chimera/issues)
2. **Search discussions**: [GitHub Discussions](https://github.com/ylab-hi/chimera/discussions)
3. **Open a new issue**: Include:
   - ChimeraLM version (`chimeralm --version`)
   - Python version (`python --version`)
   - Operating system
   - Complete error message
   - Minimal reproducible example

!!! tip "Before Opening an Issue"

    - Update to the latest version
    - Try with sample data (`tests/data/mk1c_test.bam`)
    - Include full error traceback
    - Describe what you expected vs. what happened
