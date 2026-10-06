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
