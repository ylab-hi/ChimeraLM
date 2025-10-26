# Quick Start

Get started with ChimeraLM in under 15 minutes! This tutorial will guide you through your first chimeric read prediction.

!!! info "What you'll learn"
    - How to run predictions on BAM files
    - Understanding ChimeraLM output format
    - Verifying your results

    **Time**: ~15 minutes

## Prerequisites

- ChimeraLM installed ([Installation Guide](installation.md))
- Basic command-line experience
- A BAM file to analyze (we'll provide sample data)

## Step 1: Get Sample Data

ChimeraLM includes test data in the repository. If you installed from source:

```bash
# Sample data is already available
ls tests/data/mk1c_test.bam
```

If you installed via pip, download the sample data:

```bash
# Download sample BAM file with index
wget https://github.com/ylab-hi/chimera/raw/main/tests/data/mk1c_test.bam
wget https://github.com/ylab-hi/chimera/raw/main/tests/data/mk1c_test.bam.bai

# Or using curl
curl -L -o mk1c_test.bam https://github.com/ylab-hi/chimera/raw/main/tests/data/mk1c_test.bam
curl -L -o mk1c_test.bam.bai https://github.com/ylab-hi/chimera/raw/main/tests/data/mk1c_test.bam.bai

# Verify files downloaded correctly
ls -lh mk1c_test.bam*
```

!!! tip "About the Sample Data"
    The sample file `mk1c_test.bam` contains 175 reads, in which 75 chimeric reads and 100 non-chimeric reads, subsampled from PC3 cell line (human prostate cancer) sequenced using Nanopore MinION Mk1C with whole genome amplification.

## Step 2: Run Your First Prediction

Run ChimeraLM on the sample data:

=== "CPU Mode"

    ```bash
    chimeralm predict mk1c_test.bam --gpus 0
    ```

    **Expected output**:
    ```text
    INFO     [rank: 0] Loading model from Hugging Face
    Seed set to 42
    GPU available: True (mps), used: False
    Generating train split: 75 examples [00:00, 1844.17 examples/s]
    Predicting DataLoader 0: 100%|██████████| 4/4 [00:15<00:00, 0.26it/s]
    ```

    Predictions saved to: `mk1c_test.predictions/`

=== "GPU Mode"

    ```bash
    chimeralm predict mk1c_test.bam --gpus 1 --batch-size 24
    ```

    **Expected output**:
    ```text
    INFO     [rank: 0] Loading model from Hugging Face
    Seed set to 42
    GPU available: True (mps), used: True
    Predicting DataLoader 0: 100%|██████████| 2/2 [00:03<00:00, 0.66it/s]
    ```

    Predictions saved to: `mk1c_test.predictions/`

!!! tip "GPU vs CPU Performance"
    - **CPU**: ~15 seconds for 48 SA-tagged reads (batch-size 12)
    - **GPU**: ~3 seconds for 48 SA-tagged reads (batch-size 24, 5x faster!)

## Step 3: Understand the Output

ChimeraLM creates a predictions file with one line per read:

```bash
# View predictions from first batch
head -10 mk1c_test.predictions/0_0.txt
```

**Output format** (tab-separated):

```text
read_name<TAB>label
e5f89040-2898-41d9-9ee4-3022168216f0	1
b76512a7-5a74-405b-8ac3-adde6a7ea5e1	0
5b830fb3-6bb7-42a4-ad18-142b9474ed7d	1
edab7cd5-831c-4f51-8ada-c9b4620307c1	0
...
```

**Labels**:

- **0**: Biological read (keep for analysis)
- **1**: Chimeric artifact (remove from analysis)

## Step 4: Interpret Results

Count how many reads are chimeric:

```bash
# Count chimeric reads (label 1)
cat mk1c_test.predictions/*.txt | grep -c "1$"

# Count biological reads (label 0)
cat mk1c_test.predictions/*.txt | grep -c "0$"
```

Expected results for test data:

- **Chimeric artifacts**: 55 (73.3%)
- **Biological reads**: 20 (26.7%)

Typical chimera rates for WGA data:

- **MDA (Multiple Displacement Amplification)**: 10-40%
- **PicoPLEX**: 5-20%
- **Non-WGA data**: \<1%

## Checkpoint: Verify Your Prediction Worked

✅ **Success indicators**:

- [ ] Predictions file created
- [ ] File contains tab-separated read names and labels
- [ ] Labels are 0 or 1
- [ ] Number of predictions matches input reads

!!! success "Congratulations!"
    You've successfully run your first ChimeraLM prediction! :tada:

## Next Steps

Now that you've completed the basics:

### For Analysis

**Filter your BAM file** to remove chimeric reads:

```bash
chimeralm filter mk1c_test.bam mk1c_test.predictions
```

This automatically creates:

- `mk1c_test.filtered.bam` - Unsorted filtered reads
- `mk1c_test.filtered.sorted.bam` - **Final sorted output** (use this!)
- `mk1c_test.filtered.sorted.bam.bai` - BAM index
- `mk1c_test.predictions/predictions.txt` - Consolidated predictions

For comprehensive filtering guidance including verification, troubleshooting, and batch processing, see the [Filtering BAM Files Tutorial](../tutorials/bam-filtering.md).

### For Learning

- **Optimize performance**: See [Performance Optimization](../tutorials/performance-optimization.md)
- **Integrate into pipelines**: See [Pipeline Integration](../tutorials/pipeline-integration.md)
- **Use the web interface**: See [Web Command](../reference/cli.md#web-command)

### For Development

- **Use as a library**: See [API Reference](../reference/models.md)

## Troubleshooting

Encountered an issue? Check our [Troubleshooting Guide](troubleshooting.md) for common problems and solutions.

!!! question "Need Help?"
    - :material-github: [Open an issue](https://github.com/ylab-hi/chimera/issues)
    - :material-chat: [GitHub Discussions](https://github.com/ylab-hi/chimera/discussions)
