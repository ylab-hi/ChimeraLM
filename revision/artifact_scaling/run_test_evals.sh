#!/bin/bash
set -u
R=/gpfs/projects/b1171/ylk4626/project/Chimera; cd $R
export HF_HOME=$R/tmp/hf_home HF_DATASETS_CACHE=$R/tmp/hf_datasets_eval3 TMPDIR=$R/tmp/tmp; mkdir -p $HF_DATASETS_CACHE
CK1=$R/logs/train/runs/2025-08-14_21-53-45/checkpoints/epoch_005_f1_0.8037.ckpt
CK2=$R/revision/artifact_scaling/art2x_20261007_0935/checkpoints/epoch_004_f1_0.8058.ckpt
D=$R/data/train_data/p2_765108_bulk; O=$R/revision/artifact_scaling/eval_20261008_1147; L=$R/logs/revision
for name in released art2x; do
  ck=$([ $name = released ] && echo $CK1 || echo $CK2)
  CUDA_VISIBLE_DEVICES=0 ~/.local/bin/uv run --no-sync python eval.py +experiment=hyena tags="[eval,$name]" ckpt_path=$ck data.train_data_path=$D/test.parquet data.val_data_path=$D/test.parquet data.test_data_path=$D/test.parquet data.batch_size=24 data.num_workers=8 trainer.devices=1 hydra.run.dir=$O/test_$name > $L/artifact_scaling_eval_test_$name.log 2>&1
  echo "[$(date)] test $name rc=$?" >> $L/artifact_scaling_eval_driver3.log
done
echo "[$(date)] TEST_EVAL_DONE" > $L/artifact_scaling_testeval_done.flag
