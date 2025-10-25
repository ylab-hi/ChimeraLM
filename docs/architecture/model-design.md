# Model Design

ChimeraLM model architecture and design decisions.

## Architecture Overview

ChimeraLM uses a two-stage architecture:

1. **Backbone**: HyenaDNA-small-32k for sequence encoding
2. **Head**: Binary sequence classifier for chimera detection

## HyenaDNA Backbone

### Overview

- **Model**: HyenaDNA-small-32k (256-dim embeddings)
- **Pretraining**: Genomic sequences (human genome)
- **Max length**: 32,768 tokens
- **Parameters**: ~1M parameters

### Architecture

```
DNA Sequence (A, C, G, T, N)
    ↓ Tokenization
Token IDs (1, 2, 3, 4, 0)
    ↓ Embedding Layer (256-dim)
Sequence Embeddings
    ↓ Hyena Layers
Contextualized Embeddings
    ↓ Classification Head
Logits (2 classes)
```

## Classification Head

### BinarySequenceClassifier

```python
class BinarySequenceClassifier(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=512, num_classes=2):
        - Attention pooling (sequence → single vector)
        - MLP: input_dim → hidden_dim → num_classes
        - Activation: GELU
        - Dropout: 0.1
```

### Pooling Strategy

**Attention-based pooling**:
- Learn importance weights for each position
- Aggregate sequence into fixed-size vector
- More flexible than mean/max pooling

## Training Configuration

### Loss Function

**CrossEntropyLoss**:
- Standard for classification
- Handles class imbalance naturally

### Optimizer

**AdamW**:
- Learning rate: 1e-4
- Weight decay: 0.01
- Betas: (0.9, 0.999)

### Scheduler

**ReduceLROnPlateau**:
- Monitor: val/loss
- Factor: 0.5
- Patience: 5 epochs

### Early Stopping

- Monitor: val/loss
- Patience: 10 epochs
- Mode: min

## Model Variants

### HyenaDNA (Default)

- **Pros**: Best accuracy, pretrained on genomics
- **Cons**: Slower training
- **Use when**: Accuracy is priority

### CNN

- **Pros**: Fast training, interpretable
- **Cons**: Lower accuracy
- **Use when**: Speed is priority

### Mamba

- **Pros**: State space model, efficient
- **Cons**: Experimental
- **Use when**: Exploring alternatives

## See Also

- [Architecture Overview](overview.md)
- [Models API Reference](../reference/models.md)
- [Fine-Tuning Tutorial](../tutorials/fine-tuning.md)
