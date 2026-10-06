#!/bin/bash
# Run HCC78 preprocessing for the four selected runs, two at a time (52 cores on qgpu0517).
set -u
R=/gpfs/projects/b1171/ylk4626/project/Chimera
L=$R/logs/revision
cd $R
S=revision/external/hcc78/preprocess_sample.sh
# wave 1: the test scWGA sample and the small MALBAC sample
$S sample_3_scWGA_MDA_R10.4_SRR21397273 24 > $L/hcc78_pre_sample3.log 2>&1 &
$S sample_8_scWGA_MALBAC_R9.4.1_SRR21397268 8 > $L/hcc78_pre_sample8.log 2>&1 &
wait
# wave 2: the two bulk samples
$S sample_1_bulk_WGS_R10.4_SRR21397275 24 > $L/hcc78_pre_sample1.log 2>&1 &
$S sample_2_bulk_WGS_R9.4.1_SRR21397274 24 > $L/hcc78_pre_sample2.log 2>&1 &
wait
echo "[$(date)] PREPROCESS_ALL_DONE" > $L/hcc78_pre_done.flag
