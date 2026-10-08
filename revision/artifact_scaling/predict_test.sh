#!/bin/bash
R=/gpfs/projects/b1171/ylk4626/project/Chimera; cd $R
pkill -9 -f 'run_test_evals'; pkill -9 -f 'python eval.py'; pkill -9 -f 'chimeralm predict'; sleep 3
echo "left: eval=$(pgrep -fc 'python eval.py') predict=$(pgrep -fc 'chimeralm predict')"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 HF_HOME=$R/tmp/hf_home HF_DATASETS_CACHE=$R/tmp/hf_datasets_pred_$(date +%H%M) TMPDIR=$R/tmp/tmp; mkdir -p $HF_DATASETS_CACHE
CK1=$R/logs/train/runs/2025-08-14_21-53-45/checkpoints/epoch_005_f1_0.8037.ckpt
CK2=$R/revision/artifact_scaling/art2x_20261007_0935/checkpoints/epoch_004_f1_0.8058.ckpt
D=$R/data/train_data/p2_765108_bulk; O=$R/revision/artifact_scaling/eval_20261008_1147; L=$R/logs/revision
run() { # name ckpt gpu
  CUDA_VISIBLE_DEVICES=$3 ~/.local/bin/uv run --no-sync python eval.py +experiment=hyena tags="[predtest,$1]" ckpt_path=$2 \
    +data.predict_data_path=$D/test.parquet data.train_data_path=$D/test.parquet data.val_data_path=$D/test.parquet data.test_data_path=$D/test.parquet \
    data.batch_size=24 data.num_workers=0 trainer.devices=1 hydra.run.dir=$O/predtest_$1 > $L/artifact_scaling_predtest_$1.log 2>&1
  echo "[$(date)] predtest $1 rc=$?" >> $L/artifact_scaling_predtest_driver.log
  find $O/predtest_$1/predicts -name '*.txt' -print0 | xargs -0 cat > $O/predtest_$1.predictions.txt
}
run released $CK1 0 &
run art2x $CK2 1 &
wait
echo "[$(date)] PREDTEST_DONE" > $L/artifact_scaling_predtest_done.flag
