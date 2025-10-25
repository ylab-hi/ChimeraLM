# Pipeline Integration

Integrate ChimeraLM into your bioinformatics pipelines using Bash, Nextflow, and Snakemake for reproducible, scalable analysis.

!!! info "Learning Objectives"
    By the end of this tutorial, you will be able to:

    - Integrate ChimeraLM into Bash scripts for simple automation
    - Build Nextflow pipelines with ChimeraLM filtering
    - Create Snakemake workflows for reproducible analysis
    - Handle errors and logging in production pipelines
    - Scale to large cohorts (100s-1000s of samples)

    **Prerequisites**: ChimeraLM installed, basic knowledge of Bash/Nextflow/Snakemake

    **Time**: ~45 minutes

## Integration Options

| Method | Best For | Complexity | Scalability |
|--------|----------|------------|-------------|
| **Bash Script** | Simple workflows, single machine | Low | 1-10 samples |
| **Nextflow** | Cloud/HPC, complex pipelines | Medium | 10-1000s samples |
| **Snakemake** | Reproducibility, local/cluster | Medium | 10-1000s samples |
| **WDL** | Cloud platforms (Terra, Cromwell) | Medium-High | 100s-1000s samples |

## Bash Script Integration

### Basic Pipeline

```bash
#!/bin/bash
# chimera_filter_pipeline.sh - Simple ChimeraLM filtering pipeline

set -euo pipefail  # Exit on error, undefined variables, pipe failures

# Configuration
INPUT_BAM=$1
OUTPUT_DIR=$2
GPUS=${3:-1}  # Default to 1 GPU
BATCH_SIZE=${4:-24}  # Default batch size 24

echo "ChimeraLM Filtering Pipeline"
echo "Input: $INPUT_BAM"
echo "Output: $OUTPUT_DIR"

# Create output directory
mkdir -p $OUTPUT_DIR

# Step 1: Predict chimeric reads
echo "Step 1/3: Predicting chimeric reads..."
chimeralm predict $INPUT_BAM --gpus $GPUS --batch-size $BATCH_SIZE

# Step 2: Filter BAM
echo "Step 2/3: Filtering BAM file..."
FILTERED_BAM="${OUTPUT_DIR}/$(basename ${INPUT_BAM%.bam}).filtered.bam"
chimeralm filter $INPUT_BAM ${INPUT_BAM}.predictions/ \
    --output-prediction $FILTERED_BAM

# Step 3: Generate QC report
echo "Step 3/3: Generating QC report..."
CHIMERIC=$(grep -c "1$" ${INPUT_BAM}.predictions/predictions.txt || echo "0")
BIOLOGICAL=$(grep -c "0$" ${INPUT_BAM}.predictions/predictions.txt || echo "0")
TOTAL=$((CHIMERIC + BIOLOGICAL))
CHIMERA_RATE=$(echo "scale=2; $CHIMERIC * 100 / $TOTAL" | bc)

cat > ${OUTPUT_DIR}/qc_report.txt <<EOF
ChimeraLM QC Report
===================
Input BAM: $INPUT_BAM
Output BAM: $FILTERED_BAM

Read Statistics:
  Total analyzed: $TOTAL
  Biological: $BIOLOGICAL
  Chimeric: $CHIMERIC
  Chimera rate: ${CHIMERA_RATE}%

Filtering complete: $(date)
EOF

echo "Pipeline complete! QC report: ${OUTPUT_DIR}/qc_report.txt"
```

### Usage

```bash
# Make script executable
chmod +x chimera_filter_pipeline.sh

# Run pipeline
./chimera_filter_pipeline.sh input.bam output/ 1 24

# Batch process multiple files
for bam in data/*.bam; do
    ./chimera_filter_pipeline.sh $bam output/ 1 24
done
```

### Advanced Bash Pipeline with Error Handling

```bash
#!/bin/bash
# advanced_chimera_pipeline.sh - Production-ready pipeline with error handling

set -euo pipefail

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a pipeline.log
}

error_exit() {
    log "ERROR: $1"
    exit 1
}

# Validate inputs
INPUT_BAM=${1:?Usage: $0 <input.bam> <output_dir> [gpus] [batch_size]}
OUTPUT_DIR=${2:?Output directory required}
GPUS=${3:-1}
BATCH_SIZE=${4:-24}

# Check dependencies
command -v chimeralm >/dev/null || error_exit "chimeralm not found"
command -v samtools >/dev/null || error_exit "samtools not found"

# Check input file exists
[[ -f $INPUT_BAM ]] || error_exit "Input BAM not found: $INPUT_BAM"

# Check disk space (need at least 2x input size)
INPUT_SIZE=$(du -b $INPUT_BAM | cut -f1)
REQUIRED_SPACE=$((INPUT_SIZE * 2))
AVAILABLE_SPACE=$(df --output=avail -B 1 $(dirname $OUTPUT_DIR) | tail -1)
[[ $AVAILABLE_SPACE -gt $REQUIRED_SPACE ]] || error_exit "Insufficient disk space"

log "Starting ChimeraLM pipeline"
log "Input: $INPUT_BAM ($(du -h $INPUT_BAM | cut -f1))"

# Step 1: Predict
log "Step 1/4: Running predictions..."
if chimeralm predict $INPUT_BAM --gpus $GPUS --batch-size $BATCH_SIZE 2>&1 | tee -a pipeline.log; then
    log "Predictions complete"
else
    error_exit "Prediction failed"
fi

# Step 2: Validate predictions
log "Step 2/4: Validating predictions..."
PRED_FILE="${INPUT_BAM}.predictions/predictions.txt"
[[ -f $PRED_FILE ]] || error_exit "Predictions file not found"
PRED_COUNT=$(wc -l < $PRED_FILE)
[[ $PRED_COUNT -gt 0 ]] || error_exit "No predictions generated"
log "Found $PRED_COUNT predictions"

# Step 3: Filter
log "Step 3/4: Filtering BAM..."
mkdir -p $OUTPUT_DIR
FILTERED_BAM="${OUTPUT_DIR}/$(basename ${INPUT_BAM%.bam}).filtered.bam"
if chimeralm filter $INPUT_BAM ${INPUT_BAM}.predictions/ --output-prediction $FILTERED_BAM 2>&1 | tee -a pipeline.log; then
    log "Filtering complete"
else
    error_exit "Filtering failed"
fi

# Step 4: Verify output
log "Step 4/4: Verifying output..."
samtools quickcheck $FILTERED_BAM || error_exit "Output BAM is corrupted"
ORIGINAL_COUNT=$(samtools view -c $INPUT_BAM)
FILTERED_COUNT=$(samtools view -c $FILTERED_BAM)
REMOVED_COUNT=$((ORIGINAL_COUNT - FILTERED_COUNT))
log "Removed $REMOVED_COUNT reads (${ORIGINAL_COUNT} -> ${FILTERED_COUNT})"

log "Pipeline complete! Output: $FILTERED_BAM"
```

## Nextflow Integration

### Simple Nextflow Pipeline

```groovy
// chimera_filter.nf - Nextflow pipeline for ChimeraLM filtering

nextflow.enable.dsl=2

// Parameters
params.input_bam = "input.bam"
params.output_dir = "results/"
params.gpus = 1
params.batch_size = 24

// Process: Predict chimeric reads
process predict {
    tag { bam.baseName }
    publishDir "${params.output_dir}/predictions", mode: 'copy'

    input:
    path bam

    output:
    tuple path(bam), path("${bam}.predictions/predictions.txt")

    script:
    """
    chimeralm predict ${bam} --gpus ${params.gpus} --batch-size ${params.batch_size}
    """
}

// Process: Filter BAM
process filter {
    tag { bam.baseName }
    publishDir "${params.output_dir}/filtered_bams", mode: 'copy'

    input:
    tuple path(bam), path(predictions)

    output:
    path "${bam.baseName}.filtered.bam"
    path "${bam.baseName}.filtered.bam.bai"

    script:
    """
    chimeralm filter ${bam} \$(dirname ${predictions}) --output-prediction ${bam.baseName}.filtered.bam
    """
}

// Process: QC report
process qc_report {
    tag { bam.baseName }
    publishDir "${params.output_dir}/qc", mode: 'copy'

    input:
    tuple path(bam), path(predictions)

    output:
    path "${bam.baseName}_qc.txt"

    script:
    """
    CHIMERIC=\$(grep -c '1\$' ${predictions} || echo 0)
    BIOLOGICAL=\$(grep -c '0\$' ${predictions} || echo 0)
    TOTAL=\$((CHIMERIC + BIOLOGICAL))
    RATE=\$(echo "scale=2; \$CHIMERIC * 100 / \$TOTAL" | bc)

    cat > ${bam.baseName}_qc.txt <<EOF
Sample: ${bam.baseName}
Total reads analyzed: \$TOTAL
Biological reads: \$BIOLOGICAL
Chimeric reads: \$CHIMERIC
Chimera rate: \${RATE}%
EOF
    """
}

// Workflow
workflow {
    // Read input BAMs
    bam_ch = Channel.fromPath(params.input_bam)

    // Run prediction
    predictions_ch = predict(bam_ch)

    // Filter BAMs
    filter(predictions_ch)

    // Generate QC reports
    qc_report(predictions_ch)
}
```

### Run Nextflow Pipeline

```bash
# Single sample
nextflow run chimera_filter.nf --input_bam input.bam --output_dir results/

# Multiple samples
nextflow run chimera_filter.nf --input_bam "data/*.bam" --output_dir results/

# With resource limits
nextflow run chimera_filter.nf \
    --input_bam "data/*.bam" \
    --output_dir results/ \
    --gpus 1 \
    --batch_size 32 \
    -with-report report.html \
    -with-trace
```

### Advanced Nextflow with Cluster Support

```groovy
// nextflow.config - Configuration for HPC cluster

process {
    executor = 'slurm'
    queue = 'gpu'
    memory = '32 GB'
    cpus = 4

    withName: predict {
        time = '2h'
        clusterOptions = '--gres=gpu:1'
    }

    withName: filter {
        time = '1h'
        cpus = 8
    }
}

docker {
    enabled = true
    runOptions = '--gpus all'
}
```

## Snakemake Integration

### Snakemake Workflow

```python
# Snakefile - Snakemake workflow for ChimeraLM filtering

configfile: "config.yaml"

# Sample names from input directory
SAMPLES = glob_wildcards("data/{sample}.bam").sample

rule all:
    input:
        expand("results/filtered_bams/{sample}.filtered.bam", sample=SAMPLES),
        expand("results/qc/{sample}_qc.txt", sample=SAMPLES),
        "results/summary_report.html"

rule predict:
    input:
        bam="data/{sample}.bam"
    output:
        predictions="results/predictions/{sample}.predictions/predictions.txt"
    params:
        gpus=config.get("gpus", 1),
        batch_size=config.get("batch_size", 24)
    log:
        "logs/predict/{sample}.log"
    shell:
        """
        chimeralm predict {input.bam} \
            --gpus {params.gpus} \
            --batch-size {params.batch_size} \
            2>&1 | tee {log}

        # Move predictions to output directory
        mv {input.bam}.predictions/ results/predictions/{wildcards.sample}.predictions/
        """

rule filter:
    input:
        bam="data/{sample}.bam",
        predictions="results/predictions/{sample}.predictions/predictions.txt"
    output:
        filtered_bam="results/filtered_bams/{sample}.filtered.bam",
        filtered_bai="results/filtered_bams/{sample}.filtered.bam.bai"
    log:
        "logs/filter/{sample}.log"
    shell:
        """
        chimeralm filter {input.bam} \
            results/predictions/{wildcards.sample}.predictions/ \
            --output-prediction {output.filtered_bam} \
            2>&1 | tee {log}
        """

rule qc_report:
    input:
        predictions="results/predictions/{sample}.predictions/predictions.txt"
    output:
        qc="results/qc/{sample}_qc.txt"
    shell:
        """
        CHIMERIC=$(grep -c '1$' {input.predictions} || echo 0)
        BIOLOGICAL=$(grep -c '0$' {input.predictions} || echo 0)
        TOTAL=$((CHIMERIC + BIOLOGICAL))
        RATE=$(echo "scale=2; $CHIMERIC * 100 / $TOTAL" | bc)

        cat > {output.qc} <<EOF
Sample: {wildcards.sample}
Total reads analyzed: $TOTAL
Biological reads: $BIOLOGICAL
Chimeric reads: $CHIMERIC
Chimera rate: ${{RATE}}%
EOF
        """

rule summary:
    input:
        qc=expand("results/qc/{sample}_qc.txt", sample=SAMPLES)
    output:
        report="results/summary_report.html"
    script:
        "scripts/generate_summary.py"
```

### Configuration File

```yaml
# config.yaml - Snakemake configuration

# ChimeraLM parameters
gpus: 1
batch_size: 24

# Cluster resources (if using cluster execution)
cluster:
  predict:
    mem: "32GB"
    cpus: 4
    time: "2:00:00"
    partition: "gpu"
  filter:
    mem: "16GB"
    cpus: 8
    time: "1:00:00"
```

### Run Snakemake Workflow

```bash
# Dry run to check workflow
snakemake -n

# Run locally with 4 cores
snakemake --cores 4

# Run on HPC cluster with SLURM
snakemake --cluster "sbatch -p {cluster.partition} -c {cluster.cpus} --mem={cluster.mem} -t {cluster.time}" \
    --cluster-config config.yaml \
    --jobs 10

# With Conda environment
snakemake --use-conda --cores 4

# Generate workflow diagram
snakemake --dag | dot -Tpng > workflow.png
```

## WDL Integration (Bonus)

### WDL Workflow

```wdl
# chimera_filter.wdl - WDL workflow for Terra/Cromwell

version 1.0

workflow ChimeraFilter {
    input {
        Array[File] input_bams
        Int gpus = 1
        Int batch_size = 24
    }

    scatter (bam in input_bams) {
        call Predict {
            input:
                bam = bam,
                gpus = gpus,
                batch_size = batch_size
        }

        call Filter {
            input:
                bam = bam,
                predictions = Predict.predictions
        }
    }

    output {
        Array[File] filtered_bams = Filter.filtered_bam
        Array[File] qc_reports = Predict.qc_report
    }
}

task Predict {
    input {
        File bam
        Int gpus
        Int batch_size
    }

    command <<<
        chimeralm predict ~{bam} --gpus ~{gpus} --batch-size ~{batch_size}
    >>>

    output {
        File predictions = "~{bam}.predictions/predictions.txt"
        File qc_report = "qc_report.txt"
    }

    runtime {
        docker: "chimeralm/chimeralm:latest"
        gpuCount: gpus
        memory: "32 GB"
        disks: "local-disk 100 HDD"
    }
}

task Filter {
    input {
        File bam
        File predictions
    }

    command <<<
        chimeralm filter ~{bam} $(dirname ~{predictions}) \
            --output-prediction filtered.bam
    >>>

    output {
        File filtered_bam = "filtered.bam"
    }

    runtime {
        docker: "chimeralm/chimeralm:latest"
        memory: "16 GB"
        disks: "local-disk 100 HDD"
    }
}
```

## Best Practices

### Error Handling

```bash
# Always use set -euo pipefail in Bash scripts
set -euo pipefail

# Check exit codes
if ! chimeralm predict input.bam --gpus 1; then
    echo "Prediction failed!" >&2
    exit 1
fi

# Use trap for cleanup
trap 'echo "Pipeline failed at line $LINENO"; exit 1' ERR
```

### Logging

```bash
# Log all output
exec 1> >(tee pipeline.log)
exec 2>&1

# Or per-command logging
chimeralm predict input.bam 2>&1 | tee predict.log
```

### Resource Management

```bash
# Limit parallel jobs based on available GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)
parallel -j $NUM_GPUS 'chimeralm predict {} --gpus 1' ::: data/*.bam
```

## Production Checklist

Before deploying to production:

- [ ] Test pipeline on sample data
- [ ] Implement error handling and logging
- [ ] Set resource limits (memory, time, GPUs)
- [ ] Add data validation checks
- [ ] Include QC report generation
- [ ] Document pipeline parameters
- [ ] Version control your pipeline code
- [ ] Test pipeline failure scenarios

## Next Steps

- **Performance tuning**: See [Performance Optimization](performance-optimization.md)
- **Fine-tuning integration**: See [Fine-Tuning Tutorial](fine-tuning.md)
- **CI/CD**: Integrate ChimeraLM filtering into GitHub Actions or GitLab CI

## Summary

You've learned how to:

- ✅ Integrate ChimeraLM into Bash scripts
- ✅ Build Nextflow pipelines for scalable processing
- ✅ Create Snakemake workflows for reproducibility
- ✅ Handle errors and logging in production
- ✅ Deploy to HPC clusters and cloud platforms
- ✅ Follow best practices for bioinformatics pipelines

!!! success "Pipeline Ready!"
    You're now ready to integrate ChimeraLM into production bioinformatics workflows!
