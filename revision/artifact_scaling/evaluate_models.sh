#!/bin/bash
# R3.Q9 evaluation: released (1x) vs art2x model on (a) the identical held-out test split and
# (b) the independent MinION WGA dataset (chimeric-read reduction). GPU 0/1.
# Usage: evaluate_models.sh <art2x_best_ckpt>
set -u
CK2=$1
R=/gpfs/projects/b1171/ylk4626/project/Chimera; cd $R
export HF_HOME=$R/tmp/hf_home HF_DATASETS_CACHE=$R/tmp/hf_datasets TMPDIR=$R/tmp/tmp
CK1=$R/logs/train/runs/2025-08-14_21-53-45/checkpoints/epoch_005_f1_0.8037.ckpt   # released model
D=$R/data/train_data/p2_765108_bulk
O=$R/revision/artifact_scaling/eval_$(date +%Y%m%d_%H%M); mkdir -p $O; L=$R/logs/revision
echo "released=$CK1" > $O/checkpoints.txt; echo "art2x=$CK2" >> $O/checkpoints.txt

# (a) test-split metrics (trainer.test; no predict_data_path => test mode)
for name in released art2x; do
  ck=$([ $name = released ] && echo $CK1 || echo $CK2)
  CUDA_VISIBLE_DEVICES=0 ~/.local/bin/uv run --no-sync python eval.py experiment=hyena tags="[eval,$name]" \
    ckpt_path=$ck data.test_data_path=$D/test.parquet data.batch_size=24 data.num_workers=16 \
    trainer.devices=1 hydra.run.dir=$O/test_$name > $L/artifact_scaling_eval_test_$name.log 2>&1
  echo "[$(date)] test $name rc=$?"
done

# (b) MinION WGA prediction with the art2x model (released-model predictions already exist)
CUDA_VISIBLE_DEVICES=1 ~/.local/bin/uv run --no-sync chimeralm predict data/raw/PC3_10_cells_MDA_Mk1c_dirty.bam --ckpt $CK2 -g 1 -b 12 -w 8 \
  -o $O/mk1c_art2x > $L/artifact_scaling_eval_mk1c_art2x.log 2>&1
echo "[$(date)] mk1c art2x rc=$?"
find $O/mk1c_art2x -name '0_*.txt' -print0 | sort -z | xargs -0 cat > $O/mk1c_art2x.predictions.txt
awk '{c[$2]++} END{printf "mk1c art2x: n=%d artifact=%d (%.2f%%)\n", c[0]+c[1], c[1], 100*c[1]/(c[0]+c[1])}' $O/mk1c_art2x.predictions.txt | tee $O/mk1c_summary.txt
echo "[$(date)] EVAL_DONE $O" > $L/artifact_scaling_eval_done.flag
