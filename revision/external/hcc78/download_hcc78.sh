#!/bin/bash
# HCC78 (PRJNA875576, Ni et al. 2023 CSBJ): minimal run set via NCBI prefetch -> vdb-validate -> fasterq-dump.
# sample_3 scWGA MDA R10.4 (test), sample_1 bulk R10.4, sample_2 bulk R9.4.1 (matched bulk), sample_8 MALBAC.
set -u
R=/gpfs/projects/b1171/ylk4626/project/Chimera/revision/external/hcc78
SRA=/software/sratoolkit/3.0.0/bin
cd "$R/raw" || exit
for spec in "SRR21397273 sample_3_scWGA_MDA_R10.4" "SRR21397275 sample_1_bulk_WGS_R10.4" "SRR21397268 sample_8_scWGA_MALBAC_R9.4.1" "SRR21397274 sample_2_bulk_WGS_R9.4.1"; do
  read -r run name <<< "$spec"
  echo "[$(date)] prefetch $run ($name)"
  for try in 1 2 3; do "$SRA/prefetch" --max-size u -O "$R/raw/sra" "$run" > "$R/raw/${run}.prefetch.log" 2>&1 && break; echo "[$(date)] prefetch retry $try"; sleep 30; done
  "$SRA/vdb-validate" "$R/raw/sra/$run" > "$R/raw/${run}.validate.log" 2>&1 && echo "[$(date)] vdb-validate OK $run" || echo "[$(date)] vdb-validate FAILED $run"
  "$SRA/fasterq-dump" --threads 16 --split-spot --skip-technical -O "$R/raw" -o "${name}_${run}.fastq" "$R/raw/sra/$run" > "$R/raw/${run}.dump.log" 2>&1 \
    && echo "[$(date)] fasterq-dump OK ${name}_${run}.fastq ($(stat -c %s "$R/raw/${name}_${run}.fastq") bytes)" || echo "[$(date)] fasterq-dump FAILED $run"
done
echo "[$(date)] ALL_DONE"
