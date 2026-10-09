#!/bin/bash
set -u
R=/gpfs/projects/b1171/ylk4626/project/Chimera; cd $R || exit
B=$R/revision/external/hcc78/bam; L=$R/logs/revision
RUST_LOG=info target/x86_64-unknown-linux-gnu/release/annotate \
  --cbam $B/sample_1_bulk_WGS_R10.4_SRR21397275.bam --cbam $B/sample_2_bulk_WGS_R9.4.1_SRR21397274.bam \
  --dbam $B/sample_3_scWGA_MDA_R10.4_SRR21397273.bam --dbam $B/sample_8_scWGA_MALBAC_R9.4.1_SRR21397268.bam \
  --ovr-threshold 1000 -t 32 --output-chimeric-events > $L/hcc78_annotate.log 2>&1
echo "[$(date)] ANNOTATE_DONE rc=$?" > $L/hcc78_annotate_done.flag
