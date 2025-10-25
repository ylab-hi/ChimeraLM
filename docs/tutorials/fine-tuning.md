# Fine-Tuning ChimeraLM

Learn how to fine-tune ChimeraLM on your own WGA data to improve accuracy for your specific sequencing setup.

!!! info "Learning Objectives"
    By the end of this tutorial, you will be able to:

    - Fine-tune ChimeraLM on custom BAM datasets
    - Use automatic data splitting for train/val/test sets
    - Customize training parameters (epochs, batch size, learning rate)
    - Monitor training progress with WandB
    - Evaluate model performance on test data

    **Prerequisites**: ChimeraLM installed, labeled BAM data, basic command-line experience

    **Time**: ~45 minutes (plus training time)

## Why Fine-Tune?

The pretrained ChimeraLM model works well for general WGA chimera detection, but fine-tuning on your own data can improve accuracy when:

- Your sequencing platform differs from the training data (e.g., different WGA kits)
- You have organism-specific artifacts
- You want to optimize for your specific chimera rate distribution
- You have high-quality labeled data from manual curation

## Step 1: Prepare Your Training Data

ChimeraLM requires BAM files with SA tags (supplementary alignment tags) for chimeric candidates. You also need labels (0=biological, 1=chimeric) for supervised learning.

### Data Format

Your training data should be a BAM file where reads are labeled. The labels can be:
- In a separate text file (recommended): `read_name\tlabel`
- In BAM tags (e.g., custom `CL:i` tag)

!!! tip "Minimum Data Requirements"
    - **Minimum**: 1,000 labeled reads (500 per class)
    - **Recommended**: 10,000+ labeled reads for robust training
    - **Ideal**: 50,000+ reads with balanced classes

### Example: Create Labels File

If you have manually curated labels:

```bash
# Format: read_name<TAB>label
cat > training_labels.txt <<EOF
m54329U_200919_012139/4194729/ccs	0
m54329U_200919_012139/4194826/ccs	1
m54329U_200919_012139/4194958/ccs	0
...
EOF
```

## Step 2: Basic Fine-Tuning with Auto-Split

The simplest way to fine-tune is with automatic data splitting:

=== "Auto-Split (Recommended)"

    ```bash
    # ChimeraLM automatically splits data into train/val/test (70/20/10)
    chimeralm finetune --train-data your_labeled_data.bam --epochs 50 --gpus 1
    ```

    **What happens:**
    - Training data is split: 70% train, 20% validation, 10% test
    - Model trains for 50 epochs
    - Validation loss is monitored for early stopping
    - Test evaluation runs at the end
    - Checkpoints saved to `logs/train/runs/YYYY-MM-DD_HH-MM-SS/`

=== "Custom Split Ratios"

    ```bash
    # Use custom split ratios (80% train, 15% val, 5% test)
    chimeralm finetune \
        --train-data your_labeled_data.bam \
        --epochs 50 \
        --gpus 1 \
        -r data.train_val_test_split=[0.8,0.15,0.05]
    ```

=== "Manual Split"

    ```bash
    # Provide separate train/val/test files
    chimeralm finetune \
        --train-data train.bam \
        --val-data val.bam \
        --test-data test.bam \
        --epochs 50 \
        --gpus 1
    ```

Expected output:
```text
Loading model from yangliz5/chimeralm...
Preparing data splits...
Train: 7000 reads, Val: 2000 reads, Test: 1000 reads

Epoch 1/50
━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:02:15
train_loss: 0.543, val_loss: 0.412, val_acc: 0.847

Epoch 2/50
━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:02:18
train_loss: 0.389, val_loss: 0.356, val_acc: 0.891
...
```

## Step 3: Customize Training Parameters

### Adjust Epochs and Batch Size

```bash
# Longer training with larger batches
chimeralm finetune \
    --train-data your_data.bam \
    --epochs 100 \
    --batch-size 32 \
    --gpus 1
```

!!! warning "Batch Size and GPU Memory"
    Larger batch sizes improve training speed but require more GPU memory:

    - **batch-size 12**: ~8GB GPU memory (default)
    - **batch-size 24**: ~16GB GPU memory
    - **batch-size 32**: ~24GB GPU memory

    If you get CUDA out of memory errors, reduce batch size.

### Set Random Seed for Reproducibility

```bash
# Fixed seed ensures reproducible results
chimeralm finetune \
    --train-data your_data.bam \
    --seed 42 \
    --epochs 50
```

### Skip Testing Phase

```bash
# Skip final test evaluation (faster iteration)
chimeralm finetune \
    --train-data your_data.bam \
    --epochs 50 \
    --no-test
```

## Step 4: Advanced Configuration with Hydra Overrides

ChimeraLM uses Hydra for configuration. You can override any parameter:

### Learning Rate and Optimizer

```bash
# Adjust learning rate
chimeralm finetune \
    --train-data your_data.bam \
    -r model.optimizer.lr=0.0001 \
    -r model.optimizer.weight_decay=0.001
```

### Mixed Precision Training

```bash
# Use 16-bit precision for faster training (A100/H100 GPUs)
chimeralm finetune \
    --train-data your_data.bam \
    -r trainer.precision=16-mixed
```

### Data Loading Workers

```bash
# Increase data loading parallelism
chimeralm finetune \
    --train-data your_data.bam \
    -r data.num_workers=8
```

### Multiple Overrides

```bash
# Combine multiple overrides
chimeralm finetune \
    --train-data your_data.bam \
    --epochs 100 \
    --batch-size 24 \
    -r model.optimizer.lr=0.0001 \
    -r trainer.precision=16-mixed \
    -r data.num_workers=8
```

## Step 5: Monitor Training with WandB

ChimeraLM integrates with Weights & Biases for experiment tracking.

### Enable WandB Logging

```bash
# WandB is enabled by default
chimeralm finetune --train-data your_data.bam --epochs 50

# View metrics at https://wandb.ai/your-username/chimeralm
```

### What WandB Tracks

- **Loss curves**: Training and validation loss per epoch
- **Metrics**: Accuracy, precision, recall, F1 score
- **System metrics**: GPU utilization, memory usage
- **Hyperparameters**: All configuration values
- **Model checkpoints**: Best model by validation loss

!!! tip "WandB Best Practices"
    - Create a WandB account at https://wandb.ai
    - Run `wandb login` before training
    - Use `wandb.ai` to compare multiple runs
    - Tag runs with dataset versions for organization

## Step 6: Evaluate and Use Your Fine-Tuned Model

### Find Your Checkpoint

After training, checkpoints are saved:

```bash
# Find the latest run
ls -lt logs/train/runs/

# Example output:
# drwxr-xr-x  5 user  staff  160 Oct 25 01:00 2025-10-25_01-00-00/

# Checkpoint location:
# logs/train/runs/2025-10-25_01-00-00/checkpoints/best.ckpt
```

### Use Fine-Tuned Model for Prediction

```bash
# Predict with your fine-tuned checkpoint
chimeralm predict input.bam \
    --ckpt logs/train/runs/2025-10-25_01-00-00/checkpoints/best.ckpt \
    --gpus 1
```

### Evaluate on Test Set

```bash
# Test evaluation runs automatically unless --no-test is used
# Results are printed at the end of training:

# Test Results:
# test_loss: 0.287
# test_acc: 0.923
# test_precision: 0.915
# test_recall: 0.931
# test_f1: 0.923
```

## Troubleshooting

### Overfitting

??? question "Validation loss increases while training loss decreases"

    **Symptom**: Training loss keeps decreasing but validation loss starts increasing

    **Cause**: Model is memorizing training data (overfitting)

    **Solutions**:
    ```bash
    # 1. Early stopping (enabled by default)
    # Training stops automatically when val_loss doesn't improve for 10 epochs

    # 2. Reduce model complexity (use CNN instead of HyenaDNA)
    chimeralm finetune --train-data your_data.bam --model cnn

    # 3. Increase dropout
    chimeralm finetune --train-data your_data.bam -r model.head.dropout=0.2

    # 4. Use more training data or data augmentation
    ```

### Underfitting

??? question "Both training and validation loss are high"

    **Symptom**: Loss plateaus at high values for both train and val

    **Cause**: Model is too simple or training is too short

    **Solutions**:
    ```bash
    # 1. Train longer
    chimeralm finetune --train-data your_data.bam --epochs 200

    # 2. Increase learning rate
    chimeralm finetune --train-data your_data.bam -r model.optimizer.lr=0.001

    # 3. Use larger model (HyenaDNA instead of CNN)
    chimeralm finetune --train-data your_data.bam --model hyena
    ```

### Slow Training

??? question "Training takes too long (>1 hour for 10k reads)"

    **Symptom**: Epochs are taking minutes instead of seconds

    **Cause**: GPU not being used or inefficient settings

    **Solutions**:
    ```bash
    # 1. Verify GPU is being used
    chimeralm finetune --train-data your_data.bam --gpus 1

    # Check GPU utilization:
    nvidia-smi

    # 2. Increase batch size (if you have GPU memory)
    chimeralm finetune --train-data your_data.bam --batch-size 32

    # 3. Enable mixed precision
    chimeralm finetune --train-data your_data.bam -r trainer.precision=16-mixed

    # 4. Increase data loading workers
    chimeralm finetune --train-data your_data.bam -r data.num_workers=8
    ```

### Imbalanced Classes

??? question "Model always predicts one class"

    **Symptom**: Accuracy is high but all predictions are label 0 (or all label 1)

    **Cause**: Training data has severe class imbalance

    **Solutions**:
    ```bash
    # 1. Check class balance in your data
    samtools view your_data.bam | grep "SA:Z:" | wc -l  # Total chimeric candidates

    # 2. Use class weighting (not yet supported - manual workaround below)
    # Balance your training data by oversampling minority class or undersampling majority

    # 3. Try different threshold for predictions (manual post-processing)
    ```

## Next Steps

- **Production deployment**: See [Pipeline Integration](pipeline-integration.md) for CI/CD workflows
- **Performance optimization**: See [Performance Optimization](performance-optimization.md) for speed tuning
- **API usage**: See [Models API Reference](../reference/models.md) for Python API

## Summary

You've learned how to:

- ✅ Prepare labeled BAM data for fine-tuning
- ✅ Fine-tune ChimeraLM with automatic data splitting
- ✅ Customize training parameters (epochs, batch size, learning rate)
- ✅ Monitor training with WandB
- ✅ Troubleshoot common training issues
- ✅ Use fine-tuned models for prediction

!!! success "Congratulations!"
    You're now ready to fine-tune ChimeraLM on your own WGA datasets!
