#!/bin/bash
# diagnostic: frozen model on the HCC78 bulk BAMs (what fraction of bulk chimeric reads is called artifact?)
set -u
R=/gpfs/projects/b1171/ylk4626/project/Chimera; cd $R || exit
export HF_HOME=$R/tmp/hf_home HF_DATASETS_CACHE=$R/tmp/hf_datasets TMPDIR=$R/tmp/tmp
B=$R/revision/external/hcc78/bam; P=$R/revision/external/hcc78/predict; L=$R/logs/revision
S1=sample_1_bulk_WGS_R10.4_SRR21397275; S2=sample_2_bulk_WGS_R9.4.1_SRR21397274
CUDA_VISIBLE_DEVICES=0 ~/.local/bin/uv run --no-sync chimeralm predict $B/$S1.bam -g 1 -b 12 -w 8 -o $P/$S1 > $L/hcc78_predict_s1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 ~/.local/bin/uv run --no-sync chimeralm predict $B/$S2.bam -g 1 -b 12 -w 8 -o $P/$S2 > $L/hcc78_predict_s2.log 2>&1 &
wait
cd $P && for s in "$S1" "$S2"; do find $s -name '0_*.txt' -print0 | sort -z | xargs -0 cat > $s.predictions.txt; done
echo "[$(date)] PREDICT_BULK_DONE" > $L/hcc78_predict_bulk_done.flag
