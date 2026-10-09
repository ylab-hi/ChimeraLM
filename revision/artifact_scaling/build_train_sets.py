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
import logging
import random
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


def split_ids(path: Path) -> tuple[set[str], int, int]:
    """Read IDs and artifact/genuine counts from a parquet file.

    Args:
        path: Path to parquet file.

    Returns:
        Tuple of (set of read IDs, artifact count, genuine count).

    """
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


def load_artifact_pool(support_path: str, used: set[str]) -> list[str]:
    """Load unused artifacts from support file.

    Args:
        support_path: Path to support file.
        used: Set of read IDs already used in splits.

    Returns:
        List of unused artifact read IDs.

    """
    pool = []
    with Path(support_path).open() as fh:
        for line in fh:
            name, sup = line.split()[:2]
            if sup == "0" and name not in used:
                pool.append(name)
    logger.info("unused artifact pool: %d", len(pool))
    return pool


def append_sampled_artifacts(
    writers: dict[int, pq.ParquetWriter],
    chunks_pattern: str,
    picks: dict[int, set[str]],
    schema: pa.Schema,
) -> dict[int, int]:
    """Stream chunks and append the sampled artifacts (label 1) to the open per-multiplier writers.

    Args:
        writers: Open ParquetWriter per multiplier (already holding the original train split).
        chunks_pattern: Glob pattern for chunk files (wildcard in the final path component).
        picks: Mapping of multiplier to picked read IDs.
        schema: Arrow schema for output files.

    Returns:
        Mapping of multiplier to number of appended rows.

    """
    multipliers = list(writers)
    all_picked = picks[max(multipliers)]
    counts = dict.fromkeys(multipliers, 0)
    value_set = pa.array(list(all_picked), pa.string())

    pattern = Path(chunks_pattern)
    for path in sorted(pattern.parent.glob(pattern.name)):
        cf = pq.ParquetFile(path)
        for rg in range(cf.num_row_groups):
            tbl = cf.read_row_group(rg)
            mask = pc.is_in(tbl["id"], value_set=value_set)
            sub = tbl.filter(mask)
            if sub.num_rows == 0:
                continue
            ids = sub["id"].to_pylist()
            for m in multipliers:
                keep = [i for i, rid in enumerate(ids) if rid in picks[m]]
                if not keep:
                    continue
                part = sub.take(pa.array(keep))
                part = part.set_column(0, "id", pa.array([f"{rid}|1" for rid in part["id"].to_pylist()], pa.string()))
                writers[m].write_table(part.cast(schema))
                counts[m] += part.num_rows
        logger.info("%s done", path.name)

    return counts


def main() -> None:
    """Build training sets with artifact augmentation.

    Parse arguments, load artifact pool, and create augmented training sets.
    """
    ap = argparse.ArgumentParser(
        description="Build training sets with 2x/4x WGA-artifact augmentation.",
    )
    ap.add_argument("--support", required=True, help="Support file path")
    ap.add_argument("--split-dir", required=True, help="Directory with split files")
    ap.add_argument("--chunks", required=True, help="Glob pattern for chunk files")
    ap.add_argument("--multipliers", nargs="+", type=int, default=[2, 4])
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

    split_dir = Path(args.split_dir)

    used: set[str] = set()
    for s in ("train", "validation", "test"):
        names, n_art, n_gen = split_ids(split_dir / f"{s}.parquet")
        used |= names
        logger.info("%s: artifacts %d genuine %d", s, n_art, n_gen)
        if s == "train":
            train_art, train_gen = n_art, n_gen
    logger.info("reads already used: %d", len(used))

    pool = load_artifact_pool(args.support, used)

    # Use seeded random for reproducibility
    rng = random.Random(args.seed)  # noqa: S311
    rng.shuffle(pool)
    need_max = train_art * (max(args.multipliers) - 1)
    if need_max > len(pool):
        msg = f"pool too small: need {need_max:,}, have {len(pool):,}"
        raise SystemExit(msg)
    # nested sampling: the 2x set is a subset of the 4x set
    picks = {m: set(pool[: train_art * (m - 1)]) for m in args.multipliers}

    schema = pq.ParquetFile(split_dir / "train.parquet").schema_arrow

    # 1) copy original training split verbatim
    pf = pq.ParquetFile(split_dir / "train.parquet")
    writers = {m: pq.ParquetWriter(split_dir / f"train_art{m}x.parquet", schema) for m in args.multipliers}
    counts = dict.fromkeys(args.multipliers, 0)
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg)
        for m in args.multipliers:
            writers[m].write_table(tbl)
            counts[m] += tbl.num_rows

    # 2) stream chunks, append sampled artifacts with label 1
    chunk_counts = append_sampled_artifacts(writers, args.chunks, picks, schema)
    for m, w in writers.items():
        w.close()
        counts[m] += chunk_counts[m]

    # Log final statistics
    for m in args.multipliers:
        logger.info(
            "train_art%dx.parquet: %d rows = %d genuine + %d artifacts (expected %d)",
            m,
            counts[m],
            train_gen,
            train_art * m,
            train_gen + train_art * m,
        )


if __name__ == "__main__":
    main()
