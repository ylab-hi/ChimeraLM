# Filtering BAM Files

Learn how to filter chimera artifacts induced by whole genome amplification (WGA) from BAM files using ChimeraLM, producing clean datasets for downstream analysis.

!!! info "Learning Objectives"
By the end of this tutorial, you will be able to:

```
- Run predictions on BAM files to identify chimera reads induced by WGA
- Filter BAM files to remove chimera artifacts induced by WGA
- Verify filtering results and quality metrics
- Integrate filtering into analysis pipelines
- Handle edge cases (empty predictions, all chimera induced by WGA, etc.)

**Prerequisites**: ChimeraLM installed, SAMtools installed, basic command-line experience

**Time**: ~20 minutes
```

## Get Sample Data

If you haven't already, download the sample BAM file with its index:

```bash
# Download sample BAM file with index
wget https://github.com/ylab-hi/chimera/raw/main/tests/data/mk1c_test.bam
wget https://github.com/ylab-hi/chimera/raw/main/tests/data/mk1c_test.bam.bai

# Or using curl
curl -L -o mk1c_test.bam https://github.com/ylab-hi/chimera/raw/main/tests/data/mk1c_test.bam
curl -L -o mk1c_test.bam.bai https://github.com/ylab-hi/chimera/raw/main/tests/data/mk1c_test.bam.bai

# Verify files
ls -lh mk1c_test.bam*
```

!!! info "About the Test Data"
The `mk1c_test.bam` file contains 175 reads, in which 75 chimeric reads and 100 non-chimeric reads, subsampled from **PC3 cell line** (human prostate cancer) sequenced using **Nanopore MinION Mk1C** with whole genome amplification.

!!! tip "Using Your Own Data"
This tutorial uses `mk1c_test.bam` as an example. Replace it with your own BAM file path throughout the tutorial.

## Workflow Overview

The ChimeraLM filtering workflow has three steps:

```mermaid
graph LR
    A[Input BAM] --> B[Predict]
    B --> C[Predictions]
    C --> D[Filter]
    D --> E[Filtered BAM]
    E --> F[Sort & Index]
    F --> G[Clean BAM]
```

1. **Predict**: Classify reads as biological (0) or chimeric artifact (1)
2. **Filter**: Remove chimeric artifact reads from BAM file
3. **Sort & Index**: Prepare filtered BAM for downstream tools

## Step 1: Run Predictions

First, identify chimera artifacts induced by WGA in your BAM file:

```bash
# Predict chimera artifacts induced by WGA

chimeralm predict mk1c_test.bam

# Or use GPU acceleration
chimeralm predict mk1c_test.bam --gpus 1

# Output directory: mk1c_test.predictions/
```

### Inspect Predictions

```bash
# View first 10 predictions from first batch
head mk1c_test.predictions/0_0.txt

# Output format (tab-separated):
# read_name<TAB>label
# e5f89040-2898-41d9-9ee4-3022168216f0	1
# b76512a7-5a74-405b-8ac3-adde6a7ea5e1	0
```

## Step 2: Filter BAM File

Remove chimera artifacts from your BAM file:

=== "Basic Filtering"

````
```bash
# Filter out chimera artifacts induced by WGA (label 1), keep biological reads (label 0)
chimeralm filter mk1c_test.bam mk1c_test.predictions
```

This automatically creates:
- `mk1c_test.filtered.bam` - Unsorted filtered reads
- `mk1c_test.filtered.sorted.bam` - **Final sorted output**
- `mk1c_test.filtered.sorted.bam.bai` - BAM index
- `mk1c_test.predictions/predictions.txt` - Consolidated predictions.txt
````

=== "Your Own Data"

````
```bash
# Replace with your BAM file
chimeralm filter your_data.bam your_data.predictions
```

Output: `your_data.filtered.sorted.bam`
````

### Expected Output

```bash
# Filter command output:
INFO     [rank: 0] Filtering mk1c_test.bam by predictions from mk1c_test.predictions                                                                              pylogger.py:46
INFO     [rank: 0] Writing all predictions to mk1c_test.predictions/predictions.txt                                                                               pylogger.py:46
INFO     [rank: 0] Loaded 75 predictions from mk1c_test.predictions                                                                                               pylogger.py:46
INFO     [rank: 0] Biological: 20 (26.7%), Chimera artifact: 55 (73.3%)                                                                                           pylogger.py:46
INFO     [rank: 0] Sorting mk1c_test.filtered.bam                                                                                                                 pylogger.py:46
INFO     [rank: 0] Indexing mk1c_test.filtered.sorted.bam                                                                                                         pylogger.py:46
```

**Files created:**

- `mk1c_test.filtered.sorted.bam` - Final sorted output (use this!)
- `mk1c_test.filtered.sorted.bam.bai` - Index file
- `mk1c_test.filtered.bam` - Intermediate unsorted file (can be deleted)

### Verify BAM Integrity

```bash
# Check BAM header
samtools view -H mk1c_test.filtered.sorted.bam | head

# Verify BAM is sorted
samtools quickcheck mk1c_test.filtered.sorted.bam && echo "BAM is valid"

# Check if indexed
ls mk1c_test.filtered.sorted.bam.bai && echo "BAM is indexed"
```

### Compare Quality Metrics

```bash
# Original BAM stats
samtools stats mk1c_test.bam > original_stats.txt

# Filtered BAM stats
samtools stats mk1c_test.filtered.sorted.bam > filtered_stats.txt

# Compare metrics
grep "^SN" original_stats.txt > original_summary.txt
grep "^SN" filtered_stats.txt > filtered_summary.txt

# View side-by-side
paste original_summary.txt filtered_summary.txt
```

## Step 4: Use Filtered BAM in Downstream Analysis

The filtered BAM is ready for any downstream tools:

### Variant Calling

```bash
# Call variants on clean data
bcftools mpileup -f reference.fa mk1c_test.filtered.sorted.bam | \
    bcftools call -mv -Oz -o variants.vcf.gz
```

### Structural Variant Detection

```bash
# Detect SVs with cleaner signal
sniffles -i mk1c_test.filtered.sorted.bam -v svs.vcf
```

### Genome Assembly

```bash
# Extract reads for assembly
samtools fasta mk1c_test.filtered.sorted.bam > clean_reads.fasta
flye --nano-raw clean_reads.fasta --out-dir assembly/
```

### Batch Filtering

Process multiple BAM files:

```bash
# Filter multiple files
for bam in *.bam; do
    echo "Processing $bam..."
    chimeralm predict $bam --gpus 1 -o ${bam}.predictions
    chimeralm filter $bam ${bam}.predictions/
    # Output: ${bam%.bam}.filtered.sorted.bam
done

echo "All files filtered!"
```

### Parallel Filtering

Use GNU parallel for faster processing:

```bash
# Install GNU parallel
# sudo apt-get install parallel  # Ubuntu
# brew install parallel  # macOS

# Predict in parallel
ls *.bam | parallel -j 4 'chimeralm predict {} --gpus 1 -o {}.predictions'

# Filter in parallel (creates .filtered.sorted.bam for each)
ls *.bam | parallel -j 8 'chimeralm filter {} {}.predictions'
```

## Troubleshooting

### Empty Predictions File

??? question "predictions.txt is empty or has very few reads"

````
**Symptom**: Predictions file exists but has 0-10 predictions

**Cause**: BAM file has no reads with SA tags (chimeric candidates)

**Solution**:
```bash
# Check for SA tags in your BAM file
samtools view your_data.bam | grep "SA:Z:" | wc -l

# If count is 0:
# Your BAM has no chimeric candidates (expected for non-WGA data)
# No filtering needed - your data is already clean!
```
````

### All Reads Labeled Chimeric

??? question "All predictions are label 1 (chimeric)"

````
**Symptom**: `grep -c "0$" predictions.txt` returns 0

**Cause**: Model is not working correctly or data is severely contaminated

**Solution**:
```bash
# 1. Check if using correct model
chimeralm predict your_data.bam --gpus 1  # Uses default pretrained model

# 2. Verify input data quality
samtools stats your_data.bam | grep "^SN"

# 3. Try with test data to verify model works
# Download test data first (see "Get Sample Data" section above)
chimeralm predict mk1c_test.bam --gpus 1

# 4. If test data works but yours doesn't, check data quality
# 5. If still all chimeric, contact support with your data
```
````

### Filtered BAM Same Size as Input

??? question "Filtered BAM has same number of reads as input"

````
**Symptom**: No reads were removed

**Cause**: All reads labeled as biological (label 0)

**Check**:
```bash
grep -c "1$" predictions.txt  # Should be > 0

# If 0, no chimeric reads detected (good quality data!)
```
````

### Filter Command Fails

??? question "chimeralm filter command fails with error"

````
**Common Errors**:

1. **Predictions directory not found**
   ```bash
   # Ensure predictions directory exists
   ls your_data.bam.predictions/predictions.txt
   ```

2. **BAM file corrupted**
   ```bash
   # Verify BAM integrity
   samtools quickcheck your_data.bam
   ```

3. **Insufficient disk space**
   ```bash
   # Check available space (need ~2x input BAM size)
   df -h .
   ```
````

## Best Practices

### Before Filtering

- [ ] Run predictions on test data first to verify model is working
- [ ] Backup original BAM file
- [ ] Ensure sufficient disk space (2x input BAM size)

### After Filtering

- [ ] Verify read counts match expectations
- [ ] Check BAM integrity with `samtools quickcheck`
- [ ] Compare quality metrics (original vs filtered)
- [ ] Keep predictions for reproducibility

### Production Pipelines

```bash
# Complete filtering pipeline with checks
BAM="input.bam"
PRED_DIR="${BAM}.predictions"
FILTERED="${BAM%.bam}.filtered.sorted.bam"

# Step 1: Predict
chimeralm predict $BAM --gpus 1 || { echo "Prediction failed"; exit 1; }

# Step 2: Check predictions exist
if [ ! -d "$PRED_DIR" ]; then
    echo "No predictions directory - prediction may have failed"
    exit 1
fi

# Step 3: Filter (creates .filtered.sorted.bam automatically)
chimeralm filter $BAM $PRED_DIR || { echo "Filtering failed"; exit 1; }

# Step 4: Verify output exists and is valid
if [ -f "$FILTERED" ]; then
    samtools quickcheck $FILTERED || { echo "Filtered BAM is corrupted"; exit 1; }
    echo "Filtering complete: $FILTERED"
else
    echo "Error: Filtered BAM not created"
    exit 1
fi
```

## Next Steps

- **Performance optimization**: See [Performance Optimization](performance-optimization.md) for faster filtering
- **Web Interface**: See [Web Interface](web-interface.md) for interactive filtering
- **Pipeline integration**: See [Pipeline Integration](pipeline-integration.md) for Nextflow/Snakemake workflows

## Summary

You've learned how to:

- ✅ Run predictions to identify chimeric reads
- ✅ Filter BAM files to remove chimeric artifacts
- ✅ Verify filtering results with SAMtools
- ✅ Integrate filtering into analysis pipelines
- ✅ Troubleshoot common filtering issues
- ✅ Batch process multiple BAM files

!!! success "Clean Data Ready!"
Your filtered BAM file is now ready for high-quality downstream analysis!
