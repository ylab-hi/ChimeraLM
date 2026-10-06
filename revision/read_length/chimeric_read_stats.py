"""Per-read statistics for chimeric reads in a BAM (revision R1: R1.Q2/Q3/Q4, R3.Q8/Q13).

A read is chimeric if its primary alignment is mapped, is not secondary/supplementary,
and carries an SA tag (same definition as the manuscript Methods).

For each chimeric read, writes one TSV row:
    name  read_len  mapq  n_segments  first_junction  last_junction  segments
where
    read_len        full read length including hard clips (infer_read_length)
    n_segments      1 + number of SA entries
    segments        ';'-separated "start-end" of every aligned segment in ORIGINAL read
                    orientation (0-based, half-open), sorted by start
    first_junction  smallest interior segment boundary (read coordinate); the model only
                    sees a junction if first_junction < MAX_LEN
    last_junction   largest interior segment boundary

Usage:
    uv run python revision/read_length/chimeric_read_stats.py in.bam out.tsv.gz [--threads N]
"""

from __future__ import annotations

import argparse
import gzip
import re
import sys
import time

import pysam

CIGAR_RE = re.compile(r"(\d+)([MIDNSHP=X])")
QUERY_CONSUMING = set("MIS=XH")  # H included so clipped coordinates refer to the full read


def sa_query_interval(cigar: str, strand: str, read_len: int) -> tuple[int, int]:
    """Query interval [start, end) of a supplementary alignment in original read orientation."""
    left = 0
    aligned = 0
    right = 0
    seen_aligned = False
    for n, op in CIGAR_RE.findall(cigar):
        n = int(n)
        if op in "SH":
            if seen_aligned:
                right += n
            else:
                left += n
        elif op in "MI=X":
            aligned += n
            seen_aligned = True
    start, end = left, left + aligned
    if strand == "-":
        start, end = read_len - end, read_len - start
    return start, end


def primary_query_interval(read: pysam.AlignedSegment, read_len: int) -> tuple[int, int]:
    """Query interval of the primary alignment in original read orientation."""
    cig = read.cigartuples or []
    left = 0
    for op, n in cig:
        if op in (4, 5):  # S, H
            left += n
        else:
            break
    right = 0
    for op, n in reversed(cig):
        if op in (4, 5):
            right += n
        else:
            break
    start, end = left, read_len - right
    if read.is_reverse:
        start, end = read_len - end, read_len - start
    return start, end


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bam")
    ap.add_argument("out")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    t0 = time.time()
    n_total = n_chim = 0
    with pysam.AlignmentFile(args.bam, "rb", threads=args.threads) as bam, gzip.open(args.out, "wt") as out:
        out.write("name\tread_len\tmapq\tn_segments\tfirst_junction\tlast_junction\tsegments\n")
        for read in bam.fetch(until_eof=True):
            n_total += 1
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            if not read.has_tag("SA"):
                continue
            n_chim += 1
            read_len = read.infer_read_length()
            segs = [primary_query_interval(read, read_len)]
            for entry in read.get_tag("SA").rstrip(";").split(";"):
                f = entry.split(",")
                if len(f) < 6:
                    continue
                segs.append(sa_query_interval(f[3], f[2], read_len))
            segs.sort()
            bounds = sorted({p for s, e in segs for p in (s, e)} - {0, read_len})
            first_j = bounds[0] if bounds else -1
            last_j = bounds[-1] if bounds else -1
            out.write(
                f"{read.query_name}\t{read_len}\t{read.mapping_quality}\t{len(segs)}\t{first_j}\t{last_j}\t"
                + ";".join(f"{s}-{e}" for s, e in segs)
                + "\n"
            )
            if n_chim % 1_000_000 == 0:
                print(f"{n_chim:,} chimeric / {n_total:,} records, {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    print(f"done: {n_chim:,} chimeric reads from {n_total:,} records in {time.time() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
