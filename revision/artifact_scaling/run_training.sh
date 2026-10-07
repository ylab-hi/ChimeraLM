#!/bin/bash
# R3.Q9: retrain ChimeraLM with 2x and 4x WGA-artifact reads in the training split (validation/test unchanged).
# Same recipe as the released model (experiment=hyena: hyenadna-small-32k, lr 1e-4, batch 16, bf16, seed 12345),
# capped at 8 epochs (released model's best epoch was 5), early-stopping patience 3. Runs sequentially on both GPUs.
set -u
R=/gpfs/projects/b1171/ylk4626/project/Chimera; cd $R
export HF_HOME=$R/tmp/hf_home HF_DATASETS_CACHE=$R/tmp/hf_datasets TMPDIR=$R/tmp/tmp
D=$R/data/train_data/p2_765108_bulk
L=$R/logs/revision
STAMP=$(date +%Y%m%d_%H%M)
for m in 2 4; do
  OUT=$R/revision/artifact_scaling/art${m}x_$STAMP
  echo "[$(date)] start art${m}x -> $OUT"
  ~/.local/bin/uv run --no-sync python train.py experiment=hyena tags="[hyena,art${m}x]" seed=12345 \
    data.train_data_path=$D/train_art${m}x.parquet data.val_data_path=$D/validation.parquet data.test_data_path=$D/test.parquet \
    data.batch_size=16 data.num_workers=30 \
    trainer.min_epochs=6 trainer.max_epochs=8 callbacks.early_stopping.patience=3 \
    hydra.run.dir=$OUT > $L/artifact_scaling_art${m}x.log 2>&1
  echo "[$(date)] done art${m}x rc=$?"
done
echo "[$(date)] TRAIN_ALL_DONE" > $L/artifact_scaling_done.flag
