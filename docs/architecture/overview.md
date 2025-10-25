# Architecture Overview

ChimeraLM system architecture and design principles.

## System Architecture

ChimeraLM follows a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                      CLI Interface                          │
│                  (Typer-based commands)                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
       ┌───────────────┼───────────────┐
       │               │               │
       ▼               ▼               ▼
┌──────────┐    ┌──────────┐    ┌──────────┐
│ Predict  │    │ Filter   │    │Finetune  │
│ Pipeline │    │ Pipeline │    │ Pipeline │
└────┬─────┘    └────┬─────┘    └────┬─────┘
     │               │               │
     │               │               │
     ▼               ▼               ▼
┌────────────────────────────────────────────┐
│           Data Processing Layer            │
│  (BAM Parser, Tokenizer, Data Collator)   │
└──────────────────┬─────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────┐
│            Model Layer                     │
│  (HyenaDNA + Classification Head)         │
└──────────────────┬─────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────┐
│         PyTorch Lightning                  │
│   (Training, Validation, Testing)         │
└────────────────────────────────────────────┘
```

## Core Components

### 1. CLI Layer

- **Typer-based CLI**: User-friendly command-line interface
- **Commands**: `predict`, `filter`, `finetune`
- **Argument parsing**: Type-safe with validation

### 2. Data Processing

- **BAM Parser**: Extracts reads with SA tags (chimeric candidates)
- **Tokenizer**: Converts DNA sequences to integer tokens
- **Data Collator**: Batching and padding for efficient training

### 3. Model Layer

- **Backbone**: HyenaDNA-small-32k (pretrained)
- **Head**: Binary sequence classifier with attention pooling
- **Loss**: CrossEntropyLoss
- **Optimizer**: AdamW

### 4. Training Framework

- **PyTorch Lightning**: Manages training loops, callbacks, logging
- **Hydra**: Configuration management
- **WandB**: Experiment tracking

## Design Principles

### Modularity

Each component is independently testable and replaceable:

- **Model backbones**: HyenaDNA, CNN, Mamba
- **Data sources**: BAM, FASTQ
- **Training frameworks**: Lightning for scalability

### Configurability

All parameters configurable via:

- **CLI flags**: Common parameters (e.g., `--batch-size`, `--gpus`)
- **Hydra overrides**: Advanced parameters (e.g., `-r model.optimizer.lr=0.0001`)
- **Config files**: Complete configuration in YAML

### Performance

Optimized for modern hardware:

- **GPU acceleration**: CUDA, MPS (Apple Silicon)
- **Batch processing**: Configurable batch sizes
- **Parallel data loading**: Multi-worker support

### Extensibility

Easy to extend:

- **Custom models**: Implement model interface
- **Custom data sources**: Implement DataModule
- **Custom callbacks**: PyTorch Lightning callbacks

## Data Flow

### Prediction Pipeline

```mermaid
graph LR
    A[BAM File] --> B[Read BAM]
    B --> C[Filter SA Tags]
    C --> D[Extract Sequences]
    D --> E[Tokenize]
    E --> F[Batch & Pad]
    F --> G[Model Inference]
    G --> H[Predictions]
    H --> I[Write to File]
```

### Training Pipeline

```mermaid
graph LR
    A[Labeled BAM] --> B[Load Data]
    B --> C[Split Train/Val/Test]
    C --> D[Tokenize]
    D --> E[Create Batches]
    E --> F[Training Loop]
    F --> G[Validation]
    G --> H[Checkpointing]
    H --> I[Test Evaluation]
```

## Technology Stack

- **Deep Learning**: PyTorch 2.5.1, PyTorch Lightning 2.4+
- **Configuration**: Hydra 1.3.2
- **Bioinformatics**: pysam 0.22.1, noodles (Rust)
- **Logging**: WandB, TensorBoard, Rich
- **CLI**: Typer
- **Language Model**: HyenaDNA (Hugging Face Transformers)

## See Also

- [Model Design](model-design.md) - Model architecture details
- [Data Pipeline](data-pipeline.md) - Data processing internals
- [Training Process](training.md) - Training workflow
- [Models API Reference](../reference/models.md) - API documentation
