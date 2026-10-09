"""Write a labelled parquet split (id = name|label, seq) as a minimal BAM that `chimeralm predict` accepts.

(mapped primary record with an SA tag). Alignment fields are dummies; only id and sequence are used by the model.

Usage: python parquet_to_bam.py test.parquet test.bam
"""

import logging
import sys

import pyarrow.parquet as pq
import pysam

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


src, dst = sys.argv[1], sys.argv[2]
table = pq.read_table(src, columns=["id", "seq"])
header = {"HD": {"VN": "1.6", "SO": "unsorted"}, "SQ": [{"LN": 1_000_000, "SN": "dummy"}]}
n = 0
with pysam.AlignmentFile(dst, "wb", header=header) as out:
    for rid, seq in zip(table["id"].to_pylist(), table["seq"].to_pylist(), strict=True):
        a = pysam.AlignedSegment()
        a.query_name = rid
        a.query_sequence = seq
        a.flag = 0
        a.reference_id = 0
        a.reference_start = 0
        a.mapping_quality = 60
        a.cigar = [(0, len(seq))]
        a.set_tag("SA", "dummy,1,+,1M,60,0;")
        out.write(a)
        n += 1
logger.info(f"wrote {n} reads to {dst}")
