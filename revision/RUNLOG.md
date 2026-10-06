# Revision R1 — run log

All runs on Quest `qgpu0517` (job 8732745) unless noted. Code repo `ylab-hi/ChimeraLM`, branch `revision-r1`.

## 2026-10-06 — read_length (R1.Q2, R1.Q3, R1.Q4, R3.Q8, R3.Q13)

- Commit: `d1da261`
- Driver: `revision/read_length/run_stats_20261006.sh` (nohup; no tmux on qgpu0517)
- Logs: `logs/revision/rl_{mk1c,test,p2,driver}.log`, completion flag `logs/revision/rl_done.flag`
- Inputs (read-only):
  - `data/raw/PC3_10_cells_MDA_Mk1c_dirty.bam` (WGA MinION, 13 GB)
  - `data/raw/PC3_10_cells_MDA_P2_dirty.bam` (WGA PromethION, 133 GB)
  - `data/train_data/p2_765108_bulk/test.parquet` (held-out test split, 76,512 reads)
- Outputs: `revision/read_length/out_20261006/{mk1c,p2}_chimeric_stats.tsv.gz`, `test_ids.tsv.gz`
- Commands:
  ```
  uv run --no-sync python revision/read_length/chimeric_read_stats.py <bam> <out.tsv.gz> --threads 8
  uv run --no-sync python revision/read_length/test_set_ids.py data/train_data/p2_765108_bulk/test.parquet <out.tsv.gz>
  ```
- Predictions used downstream (local, not re-run): `chimeralm_paper/figures/data/prediction_and_chimeric/hyena_p2_765108_bulk_{p2,mk1c}_predicts.txt` (class 1 = artifact; model ckpt `logs/train/runs/2025-08-14_21-53-45/checkpoints/epoch_005_f1_0.8037.ckpt`)
- CPU only; GPUs idle.
- Result (2026-10-06, 345 s for P2, 45 s for Mk1c): P2 12,963,576 chimeric reads, median 2,100 bp, 1,211 (0.009 %) > 32,768 bp, 2 with all junctions beyond 32,768; Mk1c 1,666,427, median 1,469 bp, 6 > 32,768. Test-set (WGA-derived, n = 58,636) P/R/F1 by bin in `chimeralm_paper/figures/data/revision/read_length/analysis/`. Figure → manuscript Extended Data Fig. 2 (`figures/final_figures/sf_read_length.pdf`).
- Note: TSVs were produced with the pre-fix junction definition (soft-clip boundaries counted); `analyze.py` recomputes junctions from the `segments` column, so results use the corrected definition. Script fixed in `56fde1f`.

## 2026-10-06 — external/hcc78 download (R2.Q2, R3.Q2, R3.Q3, R3.Q5)

- Dataset: PRJNA875576, Ni et al. 2023 CSBJ (doi 10.1016/j.csbj.2023.03.038), City Univ. Hong Kong. Single HCC78 cells, REPLI-g Single Cell MDA (sample_3–7) + MALBAC (sample_8), ONT MinION R10.4 (LSK112) / R9.4.1 (LSK110), Guppy 6; bulk HCC78 WGS R10.4 (sample_1) and R9.4.1 (sample_2).
- Manifest of all 8 ONT runs (ENA URLs + md5) kept for reference: `revision/external/hcc78/manifest.tsv`.
- PI decision: download only the runs needed — SRR21397273 (sample_3, scWGA MDA R10.4, test), SRR21397275 (sample_1, bulk R10.4), SRR21397274 (sample_2, bulk R9.4.1; both bulks ≈13× for labelling), SRR21397268 (sample_8, MALBAC). ≈52 GB gz.
- Transfer: ENA wget 1.4 MB/s and Aspera (auth refused) abandoned; NCBI `prefetch` (sratoolkit 3.0.0) → `vdb-validate` → `fasterq-dump --threads 16` to plain FASTQ. Command (qgpu0517, nohup): `revision/external/hcc78/download_hcc78.sh` → `revision/external/hcc78/raw/<name>_<run>.fastq`; log `logs/revision/hcc78_download.log` (ALL_DONE at end).
- Not yet run: alignment / prediction (awaiting PI approval).
- Other candidates evaluated and parked: PRJNA935844 (NA12878 MDA, PacBio CLR, 3rd-ChimeraMiner authors); EGAS50000001156 (brain dMDA ONT, controlled); SciLifeLab 10.17044/scilifelab.22730684 (T-cell dMDA HiFi, controlled).

## 2026-10-06 — residual_sv (R3.Q6, R3.Q11)

- Commit `4911567` (+ bin fix). Login node, seconds. Inputs: Truvari outputs of QG's SUPPORT≥3 benchmark (`20260401_R2Q1_.../2_cross_platform_strict_GT_truvari_benchmark/3{b,c}_right_*/truvari_output/{fp,tp-comp}.vcf.gz`).
- Command: `uv run --no-sync python revision/residual_sv/residual_sv_features.py PromethION=<3b_right> MinION=<3c_right> --out revision/residual_sv/out_20261006`
- Result: P2 unsupported 4,332 (INV 24.0 %, median 189 bp, median SUPPORT 4) vs supported 4,490 (1 INV, median SUPPORT 10); SUPPORT≥5 removes 56.3 % unsupported / 20.1 % supported; ≥10: 80.3 % / 48.6 %. Mk1c 606 vs 1,450; stricter thresholds remove both classes similarly. → Ext Data Fig 4.
