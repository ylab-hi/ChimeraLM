#!/bin/bash
# rerun of sample_3 prediction with HF caches on GPFS (home quota exceeded on first attempt)
set -u
R=/gpfs/projects/b1171/ylk4626/project/Chimera; cd "$R" || exit
export HF_HOME=$R/tmp/hf_home HF_DATASETS_CACHE=$R/tmp/hf_datasets TMPDIR=$R/tmp/tmp
mkdir -p $HF_HOME $HF_DATASETS_CACHE $TMPDIR
B=$R/revision/external/hcc78/bam; P=$R/revision/external/hcc78/predict; L=$R/logs/revision
S3=sample_3_scWGA_MDA_R10.4_SRR21397273
rm -rf "${P:?}/${S3:?}"
CUDA_VISIBLE_DEVICES=0 ~/.local/bin/uv run --no-sync chimeralm predict $B/$S3.bam -g 1 -b 12 -w 8 -o $P/$S3 > $L/hcc78_predict_s3.log 2>&1
echo "[$(date)] PREDICT_S3_DONE rc=$?" > $L/hcc78_predict_s3_done.flag
