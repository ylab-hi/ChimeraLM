"""Build training sets with 2x / 4x WGA-artifact reads (R3.Q9), streaming parquet (low memory).

Artifact pool = reads with bulk support 0 in the P2 support file, excluding every read already in
train/validation/test. Sampled artifacts are appended (id|1) to the original training split;
validation and test splits are left untouched.

Usage:
  uv run --no-sync python revision/artifact_scaling/build_train_sets.py \
      --support data/raw/PC3_10_cells_MDA_P2_dirty.threshold_1000.sup.txt \
      --split-dir data/train_data/p2_765108_bulk \
      --chunks 'data/raw/PC3_10_cells_MDA_P2_dirty.chimeric.fq_chunks/*.parquet' \
      --multipliers 2 4 --seed 12345
"""

from __future__ import annotations

import argparse
import glob
import random
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def split_ids(path: Path) -> tuple[set[str], int, int]:
    names, n_art, n_gen = set(), 0, 0
    pf = pq.ParquetFile(path)
    for rg in range(pf.num_row_groups):
        for rid in pf.read_row_group(rg, columns=["id"])["id"].to_pylist():
            name, _, label = rid.rpartition("|")
            names.add(name)
            if label == "1":
                n_art += 1
            else:
                n_gen += 1
    return names, n_art, n_gen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--support", required=True)
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--multipliers", nargs="+", type=int, default=[2, 4])
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()
    split_dir = Path(args.split_dir)

    used: set[str] = set()
    for s in ("train", "validation", "test"):
        names, n_art, n_gen = split_ids(split_dir / f"{s}.parquet")
        used |= names
        print(f"{s}: artifacts {n_art:,} genuine {n_gen:,}", file=sys.stderr)
        if s == "train":
            train_art, train_gen = n_art, n_gen
    print(f"reads already used: {len(used):,}", file=sys.stderr)

    pool = []
    with open(args.support) as fh:
        for line in fh:
            name, sup = line.split()[:2]
            if sup == "0" and name not in used:
                pool.append(name)
    print(f"unused artifact pool: {len(pool):,}", file=sys.stderr)

    rng = random.Random(args.seed)
    rng.shuffle(pool)
    need_max = train_art * (max(args.multipliers) - 1)
    if need_max > len(pool):
        raise SystemExit(f"pool too small: need {need_max:,}, have {len(pool):,}")
    # nested sampling: the 2x set is a subset of the 4x set
    picks = {m: set(pool[: train_art * (m - 1)]) for m in args.multipliers}
    all_picked = picks[max(args.multipliers)]

    schema = pq.ParquetFile(split_dir / "train.parquet").schema_arrow
    writers = {m: pq.ParquetWriter(split_dir / f"train_art{m}x.parquet", schema) for m in args.multipliers}
    counts = {m: 0 for m in args.multipliers}

    # 1) copy original training split verbatim
    pf = pq.ParquetFile(split_dir / "train.parquet")
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg)
        for m in args.multipliers:
            writers[m].write_table(tbl)
            counts[m] += tbl.num_rows

    # 2) stream chunks, append sampled artifacts with label 1
    value_set = pa.array(list(all_picked), pa.string())
    for path in sorted(glob.glob(args.chunks)):
        cf = pq.ParquetFile(path)
        for rg in range(cf.num_row_groups):
            tbl = cf.read_row_group(rg)
            mask = pc.is_in(tbl["id"], value_set=value_set)
            sub = tbl.filter(mask)
            if sub.num_rows == 0:
                continue
            ids = sub["id"].to_pylist()
            for m in args.multipliers:
                keep = [i for i, rid in enumerate(ids) if rid in picks[m]]
                if not keep:
                    continue
                part = sub.take(pa.array(keep))
                part = part.set_column(0, "id", pa.array([f"{rid}|1" for rid in part["id"].to_pylist()], pa.string()))
                writers[m].write_table(part.cast(schema))
                counts[m] += part.num_rows
        print(f"{Path(path).name} done", file=sys.stderr, flush=True)
    for m, w in writers.items():
        w.close()
        print(f"train_art{m}x.parquet: {counts[m]:,} rows = {train_gen:,} genuine + {train_art * m:,} artifacts "
              f"(expected {train_gen + train_art * m:,})", file=sys.stderr)


if __name__ == "__main__":
    main()
