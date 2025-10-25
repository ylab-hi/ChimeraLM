import pysam
import sys

bam_path = sys.argv[1]
read_names = set()
with pysam.AlignmentFile(bam_path, "rb") as bamfile:
    for read in bamfile.fetch():
        read_names.add(read.query_name)

print(f"Total reads: {len(read_names)}")
