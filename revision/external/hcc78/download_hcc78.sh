#!/bin/bash
# HCC78 (PRJNA875576, Ni et al. 2023 CSBJ) ONT FASTQ download from ENA; verifies md5.
set -u
cd /gpfs/projects/b1171/ylk4626/project/Chimera/revision/external/hcc78/raw
tail -n +2 manifest.tsv | while IFS=$'\t' read run sample strategy selection bytes md5 url; do
  out="${sample}_${run}_${strategy}_${selection}.fastq.gz"
  echo "[$(date)] start $out ($bytes bytes)"
  for try in 1 2 3; do
    wget -q -c -O "$out" "$url" && break
    echo "[$(date)] retry $try for $out"; sleep 30
  done
  got=$(md5sum "$out" | cut -d' ' -f1)
  if [ "$got" = "$md5" ]; then echo "[$(date)] OK md5 $out"; else echo "[$(date)] MD5 MISMATCH $out got=$got want=$md5"; fi
done
echo "[$(date)] ALL_DONE"
