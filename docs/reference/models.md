# Models API Reference

Python API reference for ChimeraLM model classes and components.

## Overview

ChimeraLM's model architecture consists of:

- **Factory class**: `ChimeraLM` for loading pretrained models
- **Base module**: `ClassificationLit` for PyTorch Lightning integration

---

## ChimeraLM Factory

### Usage

#### Load Pretrained Model

```python
from chimeralm.models.lm import ChimeraLM

# Load from Hugging Face Hub (default)
model = ChimeraLM.from_pretrained("yangliz5/chimeralm")

# Load from local checkpoint
model = ChimeraLM.from_pretrained("/path/to/checkpoint.ckpt")
```

#### Create New Model

```python
from chimeralm.models.lm import ChimeraLM

# Create new ChimeraLM instance
model = ChimeraLM.new(
    model_name="hyena",
    num_classes=2,
    optimizer_config={"lr": 1e-4, "weight_decay": 0.01}
)
```

### Methods

#### `from_pretrained()`

Load a pretrained ChimeraLM model.

**Parameters:**

- `model_name_or_path` (str): Hugging Face model ID or local checkpoint path
- `**kwargs`: Additional arguments passed to `ClassificationLit`

**Returns:**

- `ClassificationLit`: Loaded model ready for inference or fine-tuning

**Example:**

```python
# From Hugging Face Hub
model = ChimeraLM.from_pretrained("yangliz5/chimeralm")

# From local file
model = ChimeraLM.from_pretrained("logs/train/runs/2025-10-25/checkpoints/best.ckpt")

# With custom config
model = ChimeraLM.from_pretrained(
    "yangliz5/chimeralm",
    map_location="cpu"  # Load on CPU
)
```

#### `new()`

Create a new ChimeraLM model instance.

**Parameters:**

- `model_name` (str): Model architecture (`"hyena"`, `"cnn"`, `"mamba"`)
- `num_classes` (int): Number of output classes (default: 2)
- `optimizer_config` (dict): Optimizer configuration
- `**kwargs`: Additional model-specific arguments

**Returns:**

- `ClassificationLit`: New model instance

**Example:**

```python
# HyenaDNA model
model = ChimeraLM.new(
    model_name="hyena",
    num_classes=2,
    optimizer_config={"lr": 1e-4}
)

# CNN model
model = ChimeraLM.new(
    model_name="cnn",
    num_classes=2,
    optimizer_config={"lr": 1e-3}
)
```

---

### Training Loop

The module implements standard PyTorch Lightning hooks:

```python
import lightning as L

# Create trainer
trainer = L.Trainer(max_epochs=50, accelerator="gpu", devices=1)

# Train model
trainer.fit(model, datamodule=data_module)

# Test model
trainer.test(model, datamodule=data_module)
```

### Methods

#### `forward()`

Forward pass through the model.

**Parameters:**

- `x` (torch.Tensor): Input sequence tensor (shape: `[batch_size, seq_length]`)

**Returns:**

- `torch.Tensor`: Logits (shape: `[batch_size, num_classes]`)

**Example:**

```python
import torch

# Input: batch of 16 sequences, each 1024 tokens
x = torch.randint(0, 5, (16, 1024))

# Forward pass
logits = model(x)  # Shape: [16, 2]

# Get predictions
predictions = torch.argmax(logits, dim=-1)  # Shape: [16]
```

#### `training_step()`

Training step for one batch.

**Parameters:**

- `batch` (dict): Batch dictionary with `"input_ids"` and `"labels"`
- `batch_idx` (int): Batch index

**Returns:**

- `torch.Tensor`: Loss value

#### `validation_step()`

Validation step for one batch.

**Parameters:**

- `batch` (dict): Batch dictionary
- `batch_idx` (int): Batch index

#### `test_step()`

Test step for one batch.

**Parameters:**

- `batch` (dict): Batch dictionary
- `batch_idx` (int): Batch index

### Metrics

The module logs the following metrics:

**Training:**
- `train/loss`: Cross-entropy loss
- `train/acc`: Accuracy

**Validation:**
- `val/loss`: Validation loss
- `val/acc`: Validation accuracy
- `val/precision`: Precision score
- `val/recall`: Recall score
- `val/f1`: F1 score

**Test:**
- `test/loss`: Test loss
- `test/acc`: Test accuracy
- `test/precision`: Precision score
- `test/recall`: Recall score
- `test/f1`: F1 score

---

## Callbacks

### PredictionWriter


Custom Lightning callback for writing predictions to disk.

**Usage:**

```python
from chimeralm.models.callbacks import PredictionWriter
import lightning as L

# Create callback
writer = PredictionWriter(
    output_dir="predictions/",
    write_interval="batch"
)

# Use with trainer
trainer = L.Trainer(callbacks=[writer])
trainer.predict(model, datamodule=data_module)
```

---

## Complete Example: Inference

```python
from chimeralm.models.lm import ChimeraLM
import torch

# 1. Load pretrained model
model = ChimeraLM.from_pretrained("yangliz5/chimeralm")
model.eval()

# 2. Prepare input (tokenized sequences)
# Assume we have tokenized DNA sequences
input_ids = torch.randint(0, 5, (4, 1024))  # 4 sequences, 1024 tokens each

# 3. Run inference
with torch.no_grad():
    logits = model(input_ids)
    predictions = torch.argmax(logits, dim=-1)

# 4. Interpret predictions
for i, pred in enumerate(predictions):
    label = "Biological" if pred == 0 else "Chimeric"
    print(f"Sequence {i}: {label}")
```

**Output:**
```text
Sequence 0: Biological
Sequence 1: Chimeric
Sequence 2: Biological
Sequence 3: Biological
```

---

## See Also

- [CLI Commands](cli.md) - Command-line interface