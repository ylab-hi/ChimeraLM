"""Extract id, label and sequence length from a labelled parquet split (no sequences written).

The parquet `id` column is "<read_name>|<label>" (label 1 = WGA artifact, 0 = genuine).

Usage:
    uv run python revision/read_length/test_set_ids.py test.parquet out.tsv.gz
"""

from __future__ import annotations

import gzip
import sys

import pyarrow.compute as pc
import pyarrow.parquet as pq


def main(parquet: str, out: str) -> None:
    pf = pq.ParquetFile(parquet)
    n = 0
    with gzip.open(out, "wt") as fh:
        fh.write("name\tlabel\tseq_len\n")
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg, columns=["id", "seq"])
            ids = tbl["id"].to_pylist()
            lens = pc.utf8_length(tbl["seq"]).to_pylist()
            for rid, ln in zip(ids, lens, strict=True):
                name, _, label = rid.rpartition("|")
                fh.write(f"{name}\t{label}\t{ln}\n")
                n += 1
    print(f"wrote {n:,} rows to {out}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
