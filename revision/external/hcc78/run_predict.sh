#!/bin/bash
set -u
R=/gpfs/projects/b1171/ylk4626/project/Chimera; cd $R
B=$R/revision/external/hcc78/bam; P=$R/revision/external/hcc78/predict; L=$R/logs/revision; mkdir -p $P
S3=sample_3_scWGA_MDA_R10.4_SRR21397273; S8=sample_8_scWGA_MALBAC_R9.4.1_SRR21397268
( ~/.local/bin/dbp count-chimeric -r -t 8 $B/$S3.bam > $L/hcc78_countchim_s3.log 2>&1; ~/.local/bin/dbp count-chimeric -r -t 8 $B/$S8.bam > $L/hcc78_countchim_s8.log 2>&1 ) &
CUDA_VISIBLE_DEVICES=0 ~/.local/bin/uv run --no-sync chimeralm predict $B/$S3.bam -g 1 -b 12 -w 8 -o $P/$S3 > $L/hcc78_predict_s3.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 ~/.local/bin/uv run --no-sync chimeralm predict $B/$S8.bam -g 1 -b 12 -w 8 -o $P/$S8 > $L/hcc78_predict_s8.log 2>&1 &
wait
echo "[$(date)] PREDICT_DONE" > $L/hcc78_predict_done.flag
