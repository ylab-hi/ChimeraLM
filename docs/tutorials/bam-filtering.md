# Filtering BAM Files

Learn how to filter chimeric artifacts from BAM files using ChimeraLM predictions, producing clean datasets for downstream analysis.

!!! info "Learning Objectives"
    By the end of this tutorial, you will be able to:

    - Run predictions on BAM files to identify chimeric reads
    - Filter BAM files to remove chimeric artifacts
    - Verify filtering results and quality metrics
    - Integrate filtering into analysis pipelines
    - Handle edge cases (empty predictions, all chimeric, etc.)

    **Prerequisites**: ChimeraLM installed, SAMtools installed, basic command-line experience

    **Time**: ~20 minutes

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

1. **Predict**: Classify reads as biological (0) or chimeric (1)
2. **Filter**: Remove chimeric reads from BAM file
3. **Sort & Index**: Prepare filtered BAM for downstream tools

## Step 1: Run Predictions

First, identify chimeric reads in your BAM file:

```bash
# Predict chimeric reads
chimeralm predict input.bam --gpus 1 --batch-size 24

# Output directory: input.bam.predictions/
# Predictions file: input.bam.predictions/predictions.txt
```

### Inspect Predictions

```bash
# View first 10 predictions
head input.bam.predictions/predictions.txt

# Output format (tab-separated):
# read_name<TAB>label
# m54329U_200919_012139/4194729/ccs	0
# m54329U_200919_012139/4194826/ccs	1
```

### Check Chimera Rate

```bash
# Count chimeric reads (label 1)
CHIMERIC=$(grep -c "1$" input.bam.predictions/predictions.txt)

# Count biological reads (label 0)
BIOLOGICAL=$(grep -c "0$" input.bam.predictions/predictions.txt)

# Calculate chimera rate
echo "Chimeric: $CHIMERIC"
echo "Biological: $BIOLOGICAL"
echo "Chimera rate: $(echo "scale=2; $CHIMERIC * 100 / ($CHIMERIC + $BIOLOGICAL)" | bc)%"
```

Expected output for WGA data:
```text
Chimeric: 2341
Biological: 7659
Chimera rate: 23.41%
```

!!! info "Typical Chimera Rates"
    - **MDA (Multiple Displacement Amplification)**: 10-40%
    - **PicoPLEX**: 5-20%
    - **MALBAC**: 15-35%
    - **Non-WGA data**: <1% (expect very few chimeric reads)

## Step 2: Filter BAM File

Remove chimeric reads from your BAM file:

=== "Default Filtering"

    ```bash
    # Filter out chimeric reads (label 1), keep biological (label 0)
    chimeralm filter input.bam input.bam.predictions/ \
        --output-prediction filtered.bam
    ```

    This creates:
    - `filtered.bam`: BAM file with only biological reads (label 0)

=== "Custom Output Directory"

    ```bash
    # Specify custom output location
    chimeralm filter input.bam input.bam.predictions/ \
        --output-prediction output/clean.bam
    ```

=== "Keep Original Directory Structure"

    ```bash
    # Output to same directory as input
    chimeralm filter input.bam input.bam.predictions/ \
        --output-prediction $(dirname input.bam)/filtered.bam
    ```

### Expected Output

```bash
# Filter command output:
Reading predictions from input.bam.predictions/predictions.txt...
Found 10000 predictions (2341 chimeric, 7659 biological)
Filtering input.bam...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:15
Wrote 7659 reads to filtered.bam
Sorting filtered BAM...
Indexing filtered BAM...
Done! Filtered BAM: filtered.bam
```

## Step 3: Verify Filtering Results

### Check Read Counts

```bash
# Count reads in original BAM
ORIGINAL=$(samtools view -c input.bam)

# Count reads in filtered BAM
FILTERED=$(samtools view -c filtered.bam)

# Count reads with SA tags in original (chimeric candidates)
SA_TAGS=$(samtools view input.bam | grep -c "SA:Z:")

echo "Original reads: $ORIGINAL"
echo "Filtered reads: $FILTERED"
echo "Removed reads: $((ORIGINAL - FILTERED))"
echo "Reads with SA tags: $SA_TAGS"
```

Expected output:
```text
Original reads: 50000
Filtered reads: 47659
Removed reads: 2341
Reads with SA tags: 10000
```

!!! note "Read Count Math"
    - **Original reads**: Total reads including chimeric and non-chimeric
    - **Reads with SA tags**: Chimeric candidates analyzed by ChimeraLM
    - **Removed reads**: Chimeric reads (label 1) from SA-tagged reads
    - **Filtered reads**: Original - Removed

### Verify BAM Integrity

```bash
# Check BAM header
samtools view -H filtered.bam | head

# Verify BAM is sorted
samtools quickcheck filtered.bam && echo "BAM is valid"

# Check if indexed
ls filtered.bam.bai && echo "BAM is indexed"
```

### Compare Quality Metrics

```bash
# Original BAM stats
samtools stats input.bam > original_stats.txt

# Filtered BAM stats
samtools stats filtered.bam > filtered_stats.txt

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
bcftools mpileup -f reference.fa filtered.bam | \
    bcftools call -mv -Oz -o variants.vcf.gz
```

### Structural Variant Detection

```bash
# Detect SVs with cleaner signal
sniffles -i filtered.bam -v svs.vcf
```

### Genome Assembly

```bash
# Extract reads for assembly
samtools fasta filtered.bam > clean_reads.fasta
flye --nano-raw clean_reads.fasta --out-dir assembly/
```

## Advanced Filtering

### Filter by Prediction Threshold

If you want more control, manually filter by confidence scores:

```python
# Custom filtering script (hypothetical - ChimeraLM outputs only labels)
import pysam

# Read predictions (assume you have confidence scores)
predictions = {}
with open("predictions.txt") as f:
    for line in f:
        read_name, label, score = line.strip().split("\t")
        predictions[read_name] = float(score)

# Filter by custom threshold
threshold = 0.8
with pysam.AlignmentFile("input.bam") as infile:
    with pysam.AlignmentFile("custom_filtered.bam", "wb", template=infile) as outfile:
        for read in infile:
            if read.query_name in predictions:
                if predictions[read.query_name] < threshold:  # Keep if score < threshold
                    outfile.write(read)
            else:
                outfile.write(read)  # Keep reads not in predictions
```

!!! warning "Current Limitation"
    ChimeraLM currently outputs only binary labels (0/1), not confidence scores. Threshold filtering is a future enhancement.

### Batch Filtering

Process multiple BAM files:

```bash
# Filter multiple files
for bam in *.bam; do
    echo "Processing $bam..."
    chimeralm predict $bam --gpus 1
    chimeralm filter $bam ${bam}.predictions/ --output-prediction filtered_${bam}
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
ls *.bam | parallel -j 4 'chimeralm predict {} --gpus 1'

# Filter in parallel
ls *.bam | parallel -j 8 'chimeralm filter {} {}.predictions/ --output-prediction filtered_{}'
```

## Troubleshooting

### Empty Predictions File

??? question "predictions.txt is empty or has very few reads"

    **Symptom**: Predictions file exists but has 0-10 predictions

    **Cause**: BAM file has no reads with SA tags (chimeric candidates)

    **Solution**:
    ```bash
    # Check for SA tags
    samtools view input.bam | grep "SA:Z:" | wc -l

    # If count is 0:
    # Your BAM has no chimeric candidates (expected for non-WGA data)
    # No filtering needed - your data is already clean!
    ```

### All Reads Labeled Chimeric

??? question "All predictions are label 1 (chimeric)"

    **Symptom**: `grep -c "0$" predictions.txt` returns 0

    **Cause**: Model is not working correctly or data is severely contaminated

    **Solution**:
    ```bash
    # 1. Check if using correct model
    chimeralm predict input.bam --gpus 1  # Uses default pretrained model

    # 2. Verify input data quality
    samtools stats input.bam | grep "^SN"

    # 3. Try with test data
    chimeralm predict tests/data/mk1c_test.sort.bam --gpus 1

    # 4. If still all chimeric, contact support with your data
    ```

### Filtered BAM Same Size as Input

??? question "Filtered BAM has same number of reads as input"

    **Symptom**: No reads were removed

    **Cause**: All reads labeled as biological (label 0)

    **Check**:
    ```bash
    grep -c "1$" predictions.txt  # Should be > 0

    # If 0, no chimeric reads detected (good quality data!)
    ```

### Filter Command Fails

??? question "chimeralm filter command fails with error"

    **Common Errors**:

    1. **Predictions directory not found**
       ```bash
       # Ensure predictions directory exists
       ls input.bam.predictions/predictions.txt
       ```

    2. **BAM file corrupted**
       ```bash
       # Verify BAM integrity
       samtools quickcheck input.bam
       ```

    3. **Insufficient disk space**
       ```bash
       # Check available space
       df -h .
       ```

## Best Practices

### Before Filtering

- [ ] Run predictions on test data first to verify model is working
- [ ] Check chimera rate is reasonable (10-40% for WGA data)
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
FILTERED="filtered_${BAM}"

# Step 1: Predict
chimeralm predict $BAM --gpus 1 || { echo "Prediction failed"; exit 1; }

# Step 2: Check predictions
PRED_COUNT=$(wc -l < ${PRED_DIR}/predictions.txt)
if [ $PRED_COUNT -eq 0 ]; then
    echo "No predictions generated - input has no chimeric candidates"
    exit 0
fi

# Step 3: Filter
chimeralm filter $BAM $PRED_DIR --output-prediction $FILTERED || { echo "Filtering failed"; exit 1; }

# Step 4: Verify
samtools quickcheck $FILTERED || { echo "Filtered BAM is corrupted"; exit 1; }

echo "Filtering complete: $FILTERED"
```

## Next Steps

- **Pipeline integration**: See [Pipeline Integration](pipeline-integration.md) for Nextflow/Snakemake workflows
- **Performance optimization**: See [Performance Optimization](performance-optimization.md) for faster filtering
- **Quality control**: See [Architecture > Data Pipeline](../architecture/data-pipeline.md) for filtering internals

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
