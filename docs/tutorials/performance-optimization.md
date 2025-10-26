# Performance Optimization

Maximize ChimeraLM's throughput and minimize prediction time with GPU acceleration, batch tuning, and parallelization strategies.

!!! info "Learning Objectives"
    By the end of this tutorial, you will be able to:

    - Choose optimal hardware (CPU, CUDA GPU, MPS) for your workload
    - Tune batch size for maximum throughput without OOM errors
    - Optimize data loading with worker processes
    - Profile and benchmark ChimeraLM performance
    - Scale to large datasets (millions of reads)

    **Prerequisites**: ChimeraLM installed, basic command-line experience

    **Time**: ~30 minutes

## Performance Overview

ChimeraLM's prediction speed depends on:

1. **Hardware**: GPU >>> CPU (10-50x speedup)
2. **Batch size**: Larger batches = better GPU utilization
3. **Data loading**: Parallel workers reduce I/O bottlenecks
4. **Dataset size**: Amortized overhead for large files

!!! tip "Key Takeaways"
    - GPU is 10-50x faster than CPU
    - Batch size of 24-64 is optimal for modern GPUs
    - Diminishing returns after batch size 64

## Step 1: Choose Your Hardware

### Check Available Hardware

```bash
# Check CUDA GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check MPS (Apple Silicon) availability
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"

# Check GPU details (if CUDA available)
nvidia-smi
```

### Hardware Recommendations

=== "NVIDIA GPU (Recommended)"

    **Best for**: Large-scale prediction (>100K reads)

    ```bash
    # Use CUDA with optimal batch size
    chimeralm predict input.bam --gpus 1 --batch-size 24
    ```

    **GPU Requirements**:

    - **Minimum**: 8GB VRAM (batch-size 12)
    - **Recommended**: 16GB VRAM (batch-size 24-32)
    - **Optimal**: 24GB+ VRAM (batch-size 48-64)

=== "Apple Silicon (M1/M2/M3)"

    **Best for**: Mac users, moderate datasets (<100K reads)

    ```bash
    # MPS is auto-detected and enabled
    chimeralm predict input.bam --gpus 1 --batch-size 12
    ```

    !!! warning "MPS Limitations"
        - Single device only (no multi-GPU)
        - Slower than CUDA GPUs
        - Limited VRAM (8-96GB depending on model)

=== "CPU"

    **Best for**: Small datasets (<10K reads), no GPU available

    ```bash
    # CPU mode with multiple workers
    chimeralm predict input.bam --workers 8
    ```

    !!! tip "CPU Optimization"
        Set `--workers` to number of CPU cores for parallelism

## Step 2: Optimize Batch Size

Batch size is the most important parameter for GPU performance.

### Finding Optimal Batch Size

```bash
# Start with default (batch-size 12)
chimeralm predict input.bam --gpus 1 --batch-size 12

# Increase until you get CUDA out of memory error
chimeralm predict input.bam --gpus 1 --batch-size 24
chimeralm predict input.bam --gpus 1 --batch-size 32
chimeralm predict input.bam --gpus 1 --batch-size 48  # May OOM on smaller GPUs
```

### Batch Size Guidelines

| GPU VRAM | Recommended Batch Size | Max Batch Size |
| -------- | ---------------------- | -------------- |
| 8GB      | 12                     | 16             |
| 12GB     | 16                     | 24             |
| 16GB     | 24                     | 32             |
| 24GB     | 32                     | 48             |
| 40GB+    | 48                     | 64+            |

!!! warning "Out of Memory Errors"
    If you get `RuntimeError: CUDA out of memory`:

    ```bash
    # Reduce batch size
    chimeralm predict input.bam --gpus 1 --batch-size 12

    # Or use CPU mode
    chimeralm predict input.bam
    ```

### Measure Throughput

```bash
# Benchmark with different batch sizes
time chimeralm predict tests/data/mk1c_test.bam --gpus 1 --batch-size 12
time chimeralm predict tests/data/mk1c_test.bam --gpus 1 --batch-size 24
time chimeralm predict tests/data/mk1c_test.bam --gpus 1 --batch-size 32

# Compare total time
```

## Step 3: Optimize Data Loading

### Increase Worker Threads

```bash
# CPU mode: Use multiple workers for parallelism
chimeralm predict input.bam --gpus 0 --workers 8

# GPU mode: 2-4 workers to keep GPU fed
chimeralm predict input.bam --gpus 1 --batch-size 24 --workers 4
```

!!! info "Worker Guidelines"
    - **CPU mode**: Set workers = number of CPU cores
    - **GPU mode**: 2-4 workers (more doesn't help)
    - **Default**: 0 (main thread only)

### I/O Bottlenecks

For very large BAM files (>10GB), I/O can become a bottleneck:

```bash
# Check if I/O is the bottleneck
# Monitor GPU utilization with nvidia-smi

# If GPU utilization < 80%, increase workers
chimeralm predict large_file.bam --gpus 1 --workers 4

# If still low, data loading is the bottleneck (not much you can do)
```

## Step 4: Scale to Large Datasets

### Multi-GPU Prediction (Advanced)

Multi-GPU prediction is currently supported. You can process different files on multiple GPUs:

```bash
chimeralm predict file1.bam --gpus 2
```

### Process in Chunks

For very large datasets, process in chunks to avoid memory issues:

```bash
# Process first 100K reads
chimeralm predict huge_file.bam --max-sample 100000 --gpus 1 --batch-size 32

# Split BAM file into chunks (manual approach)
samtools view -h huge_file.bam | head -100000 | samtools view -Sb > chunk1.bam
samtools view -h huge_file.bam | tail -100000 | samtools view -Sb > chunk2.bam

chimeralm predict chunk1.bam --gpus 1 --batch-size 32
chimeralm predict chunk2.bam --gpus 1 --batch-size 32
```

## Step 5: Profile and Benchmark

### Measure End-to-End Time

```bash
# Time the full pipeline
time chimeralm predict input.bam --gpus 1 --batch-size 24

# Output:
# real    0m29.123s
# user    0m45.678s
# sys     0m3.456s
```

### Monitor GPU Utilization

```bash
# Run nvidia-smi in another terminal while predicting
watch -n 1 nvidia-smi

# Check GPU utilization (should be 90-100% during inference)
# Check GPU memory usage
```

## Best Practices

### For Maximum Speed

```bash
# NVIDIA GPU: Large batch, mixed precision
chimeralm predict input.bam --gpus 1 --batch-size 48 --workers 4

# Apple Silicon: Moderate batch
chimeralm predict input.bam --gpus 1 --batch-size 12 --workers 2

# CPU: Multiple workers
chimeralm predict input.bam --gpus 0 --workers 8 --batch-size 32
```

### For Limited GPU Memory

```bash
# Small batch size
chimeralm predict input.bam --gpus 1 --batch-size 8
```

### For Reproducibility

```bash
# Fixed seed, single worker
chimeralm predict input.bam --gpus 1 --batch-size 24 --workers 0
```

## Troubleshooting

### Slow Predictions

??? question "Predictions are slower than expected"

    **Symptom**: GPU mode is not much faster than CPU

    **Possible Causes**:

    1. **GPU not being used**
        ```bash
        # Check GPU is detected
        python -c "import torch; print(torch.cuda.is_available())"

        # Verify --gpus 1 flag is set
        chimeralm predict input.bam --gpus 1
        ```

    2. **Batch size too small**
        ```bash
        # Increase batch size
        chimeralm predict input.bam --gpus 1 --batch-size 32
        ```

    3. **I/O bottleneck**
        ```bash
        # Increase workers
        chimeralm predict input.bam --gpus 1 --workers 4
        ```

### Out of Memory

??? question "CUDA out of memory error"

    **Solutions**:

    ```bash
    # 1. Reduce batch size
    chimeralm predict input.bam --gpus 1 --batch-size 12

    # 2. Use CPU mode
    chimeralm predict input.bam --gpus 0
    ```

### GPU Not Detected

??? question "GPU available but not being used"

    **Check CUDA installation**:

    ```bash
    # Verify CUDA is available to PyTorch
    python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"

    # If False, reinstall PyTorch with CUDA
    pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
    ```

## Performance Checklist

Before running large-scale predictions:

- [ ] GPU is detected (`python -c "import torch; print(torch.cuda.is_available())"`)
- [ ] Optimal batch size determined (start with 24, increase until OOM)
- [ ] Workers set appropriately (2-4 for GPU, 8+ for CPU)
- [ ] GPU utilization monitored with `nvidia-smi`
- [ ] Benchmark completed on sample data

## Next Steps

- **Web Interface**: See [Web Interface](web-interface.md) for interactive filtering
- **Pipeline integration**: See [Pipeline Integration](pipeline-integration.md) for scaling across multiple samples

## Summary

You've learned how to:

- ✅ Choose optimal hardware for your workload
- ✅ Tune batch size for maximum throughput
- ✅ Optimize data loading with worker processes
- ✅ Scale to large datasets with chunking
- ✅ Profile and benchmark performance
- ✅ Troubleshoot common performance issues

!!! success "Performance Boost Achieved!"
    With proper optimization, you can achieve 10-50x speedup compared to default settings!
