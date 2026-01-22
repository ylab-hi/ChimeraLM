# ChimeraLM

<div class="hero" markdown>

## Genomic Language Model for Detecting WGA Chimeric Artifacts

A deep learning-powered tool to identify artificial chimeric reads arising from whole genome amplification (WGA) processes.

[Get Started](getting-started/quick-start.md){ .md-button .md-button--primary }
[Try Web Demo :material-open-in-new:](https://huggingface.co/spaces/yangliz5/ChimeraLM){ .md-button }
[View on GitHub](https://github.com/ylab-hi/chimera){ .md-button }

</div>

______________________________________________________________________

## :material-star: Key Features

<div class="feature-grid" markdown>

<div class="feature-item" markdown>
### :material-speedometer: High Accuracy
Deep learning model trained on real WGA data for precise chimeric artifact detection
</div>

<div class="feature-item" markdown>
### :material-flash: GPU Accelerated
Optimized for CUDA, MPS (Apple Silicon), and CPU with configurable batch processing
</div>

<div class="feature-item" markdown>
### :material-console-line: Easy to Use
Simple CLI with sensible defaults - get started in minutes
</div>

<div class="feature-item" markdown>
### :material-gauge: Fast Processing
Batch inference with configurable parallelism for large-scale genomic datasets
</div>

<div class="feature-item" markdown>
### :material-web: Web Interface
[Try the interactive demo](https://huggingface.co/spaces/yangliz5/ChimeraLM) on HuggingFace Spaces - no installation needed!
</div>

<div class="feature-item" markdown>
### :material-check-circle: Production Ready
Includes filtering, sorting, and indexing of BAM files
</div>

</div>

______________________________________________________________________

## Quick Start

Get up and running with ChimeraLM in under 15 minutes:

```bash
# Install ChimeraLM
pip install chimeralm

# Predict chimeric reads (CPU)
chimeralm predict your_data.bam

# Predict with GPU acceleration
chimeralm predict your_data.bam --gpus 1 --batch-size 24
```

Ready to dive in? Check out our [Quick Start Guide](getting-started/quick-start.md).

!!! tip "Try ChimeraLM Online - No Installation Required!"

    Want to test ChimeraLM before installing? Try our interactive web demo:

    **[:material-rocket-launch: Launch Web Demo on HuggingFace Spaces](https://huggingface.co/spaces/yangliz5/ChimeraLM){ .md-button .md-button--primary }**

    Perfect for:

    - :material-test-tube: Testing with individual DNA sequences
    - :material-chart-bar: Visualizing prediction confidence scores
    - :material-school: Learning about chimeric artifact detection
    - :material-flash: Quick validation before batch processing

    The web demo runs the same model as the CLI tool but provides an intuitive visual interface for single-sequence analysis.

______________________________________________________________________

## What is ChimeraLM?

ChimeraLM is a genomic language model that detects chimeric artifacts introduced by whole genome amplification (WGA). Built with PyTorch Lightning and optimized for modern GPUs, it provides fast and accurate identification of chimeric reads in BAM files.

**Chimeric artifacts** are artificial DNA sequences created during WGA that combine sequences from different genomic locations. These artifacts can lead to incorrect biological conclusions if not removed from analysis.

ChimeraLM uses the HyenaDNA backbone architecture to learn patterns that distinguish biological reads (label 0) from chimeric artifacts (label 1), helping researchers clean their sequencing data before downstream analysis.

______________________________________________________________________

## Related Tools

- **[DeepChopper](https://ylab-hi.github.io/DeepChopper/)** - For identifying chimeric reads caused by internal adapter sequences in Nanopore direct RNA sequencing (dRNA-seq) data

______________________________________________________________________

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

______________________________________________________________________

## License

ChimeraLM is licensed under the Apache License 2.0. See [License](about/license.md) for details.
