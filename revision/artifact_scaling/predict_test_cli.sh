#!/bin/bash
R=/gpfs/projects/b1171/ylk4626/project/Chimera; cd "$R" || exit
pkill -9 -f 'predict_test.sh'; pkill -9 -f 'python eval.py'; sleep 2
HF_DATASETS_CACHE="$R/tmp/hf_datasets_cli_$(date +%H%M)"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 HF_HOME="$R/tmp/hf_home" HF_DATASETS_CACHE="$HF_DATASETS_CACHE" TMPDIR="$R/tmp/tmp"
mkdir -p "$HF_DATASETS_CACHE"
CK1=$R/logs/train/runs/2025-08-14_21-53-45/checkpoints/epoch_005_f1_0.8037.ckpt
CK2=$R/revision/artifact_scaling/art2x_20261007_0935/checkpoints/epoch_004_f1_0.8058.ckpt
O=$R/revision/artifact_scaling/eval_20261008_1147; L=$R/logs/revision
[ -s $O/test.bam ] || ~/.local/bin/uv run --no-sync python revision/artifact_scaling/parquet_to_bam.py data/train_data/p2_765108_bulk/test.parquet $O/test.bam
run() { # name ckpt gpu
  rm -rf "$O/clitest_$1"
  CUDA_VISIBLE_DEVICES="$3" ~/.local/bin/uv run --no-sync chimeralm predict "$O/test.bam" --ckpt "$2" -g 1 -b 12 -w 4 -o "$O/clitest_$1" > "$L/artifact_scaling_clitest_$1.log" 2>&1
  echo "[$(date)] clitest $1 rc=$?" >> $L/artifact_scaling_clitest_driver.log
  find "$O/clitest_$1" -name '*.txt' -print0 | xargs -0 cat > "$O/clitest_$1.predictions.txt"
}
run released $CK1 0 &
run art2x $CK2 1 &
wait
echo "[$(date)] CLITEST_DONE" > $L/artifact_scaling_clitest_done.flag
