#!/bin/bash
# HCC78 (PRJNA875576) preprocessing, mirroring the PC3 pipeline:
#   chopper -q 10 -l 500  ->  porechop_abi -abi  ->  cutadapt (LSK adapters, 2 passes, e=0.1)
#   -> minimap2 -ax map-ont --MD -Y (GRCh38.p13)  ->  samtools sort/index
# Usage: preprocess_sample.sh <name> <threads>      e.g. preprocess_sample.sh sample_3_scWGA_MDA_R10.4_SRR21397273 24
set -euo pipefail
NAME=$1; T=${2:-16}
R=/gpfs/projects/b1171/ylk4626/project/Chimera/revision/external/hcc78
E=/projects/b1171/ylk4626/mambaforge/envs/deepchopper/bin
CUTADAPT=$HOME/.local/bin/cutadapt
MMI=/home/qgn1237/qgn1237/1_my_database/GRCh38_p13/minimap2_index/GRCh38.p13.genome.mmi   # same GRCh38.p13 index as PC3
IN=$R/raw/${NAME}.fastq
W=$R/work/$NAME; mkdir -p "$W" "$R/bam"
log() { echo "[$(date '+%F %T')] $NAME: $*"; }

log "start (threads=$T) $(stat -c %s "$IN") bytes"
log "chopper"
"$E"/chopper -q 10 -l 500 --threads "$T" < "$IN" > "$W/q10l500.fastq"
log "chopper done: $(awk 'NR%4==2{n++} END{print n}' "$W/q10l500.fastq") reads"
log "porechop_abi"
"$E"/porechop_abi -abi -t "$T" -i "$W/q10l500.fastq" -o "$W/porechop.fastq" > "$W/porechop.log" 2>&1
log "cutadapt"
$CUTADAPT -g TTTTTTTTCCTGTACTTCGTTCAGTTACGTATTGCT -a AGCAATACGTAACTGAACGAAGTACAGGAAAAAAAA \
          -g GCAATACGTAACTGAACGAAGTACAGG -a CCTGTACTTCGTTCAGTTACGTATTGC \
          -j "$T" --error-rate=0.1 --times=2 -o "$W/trimmed.fastq" "$W/porechop.fastq" > "$W/cutadapt.log" 2>&1
log "minimap2"
"$E"/minimap2 -ax map-ont --MD -t "$T" -Y -R "@RG\tID:${NAME}\tPL:ont\tLB:library\tSM:HCC78" "$MMI" "$W/trimmed.fastq" \
  | "$E"/samtools sort -@ 4 -m 2G -O BAM -o "$R/bam/${NAME}.bam" -
"$E"/samtools index "$R/bam/${NAME}.bam"
"$E"/samtools flagstat -@ 4 "$R/bam/${NAME}.bam" > "$R/bam/${NAME}.flagstat.txt"
log "done -> $R/bam/${NAME}.bam"
rm -f "$W/q10l500.fastq" "$W/porechop.fastq"   # keep trimmed.fastq (final reads)
