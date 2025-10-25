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
ls tests/data/mk1c_test.sort.bam
```

If you installed via pip, download the sample data:

```bash
# Download sample BAM file with index
wget https://github.com/ylab-hi/chimera/raw/main/tests/data/mk1c_test.sort.bam
wget https://github.com/ylab-hi/chimera/raw/main/tests/data/mk1c_test.sort.bam.bai

# Or using curl
curl -L -o mk1c_test.sort.bam https://github.com/ylab-hi/chimera/raw/main/tests/data/mk1c_test.sort.bam
curl -L -o mk1c_test.sort.bam.bai https://github.com/ylab-hi/chimera/raw/main/tests/data/mk1c_test.sort.bam.bai

# Verify files downloaded correctly
ls -lh mk1c_test.sort.bam*
```

!!! tip "About the Sample Data"
    The sample file `mk1c_test.sort.bam` contains 1000 reads with SA tags (chimeric candidates) subsampled from PC3 cell line sequenced by Nanopore MinION Mk1C. It's perfect for testing ChimeraLM predictions.

## Step 2: Run Your First Prediction

Run ChimeraLM on the sample data:

=== "CPU Mode"

    ```bash
    chimeralm predict mk1c_test.sort.bam --gpus 0
    ```

    **Expected output**:
    ```text
    Loading model from yangliz5/chimeralm...
    Processing BAM file: mk1c_test.sort.bam
    Found 1000 reads with SA tags (chimeric candidates)
    Running predictions on CPU...
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:45
    Predictions saved to: mk1c_test.sort.bam.predictions/predictions.txt
    ```

=== "GPU Mode"

    ```bash
    chimeralm predict mk1c_test.sort.bam --gpus 1 --batch-size 24
    ```

    **Expected output**:
    ```text
    Loading model from yangliz5/chimeralm...
    GPU detected: NVIDIA GeForce RTX 3090
    Processing BAM file: mk1c_test.sort.bam
    Found 1000 reads with SA tags (chimeric candidates)
    Running predictions on GPU...
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:05
    Predictions saved to: mk1c_test.sort.bam.predictions/predictions.txt
    ```

!!! tip "GPU vs CPU Performance"
    - **CPU**: ~45 seconds for 1000 reads
    - **GPU**: ~5 seconds for 1000 reads (10x faster!)

## Step 3: Understand the Output

ChimeraLM creates a predictions file with one line per read:

```bash
# View predictions
head -10 mk1c_test.sort.bam.predictions/predictions.txt
```

**Output format** (tab-separated):
```text
read_name       label
m54329U_200919_012139/4194729/ccs   0
m54329U_200919_012139/4194826/ccs   1
m54329U_200919_012139/4194958/ccs   0
m54329U_200919_012139/4195088/ccs   1
...
```

**Labels**:
- **0**: Biological read (keep for analysis)
- **1**: Chimeric artifact (remove from analysis)

## Step 4: Interpret Results

Count how many reads are chimeric:

```bash
# Count chimeric reads (label 1)
grep -c "1$" mk1c_test.sort.bam.predictions/predictions.txt

# Count biological reads (label 0)
grep -c "0$" mk1c_test.sort.bam.predictions/predictions.txt
```

Typical results for WGA data:
- **10-30%** chimeric artifacts (label 1)
- **70-90%** biological reads (label 0)

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
chimeralm filter mk1c_test.sort.bam mk1c_test.sort.bam.predictions/
```

This creates:
- `mk1c_test.sort.filtered.bam` - Filtered reads (unsorted)
- `mk1c_test.sort.filtered.sorted.bam` - Sorted and indexed final output

For comprehensive filtering guidance including verification, troubleshooting, and batch processing, see the [Filtering BAM Files Tutorial](../tutorials/bam-filtering.md).

### For Learning

- **Optimize performance**: See [Performance Optimization](../tutorials/performance-optimization.md)
- **Integrate into pipelines**: See [Pipeline Integration](../tutorials/pipeline-integration.md)
- **Use the web interface**: See [Web Command](../reference/cli.md#web-command)

### For Development

- **Use as a library**: See [API Reference](../reference/models.md)
- **Understand the architecture**: See [Architecture Overview](../architecture/overview.md)

## Troubleshooting

Encountered an issue? Check our [Troubleshooting Guide](troubleshooting.md) for common problems and solutions.

!!! question "Need Help?"
    - :material-github: [Open an issue](https://github.com/ylab-hi/chimera/issues)
    - :material-chat: [GitHub Discussions](https://github.com/ylab-hi/chimera/discussions)
