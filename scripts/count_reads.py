import pysam
import sys

bam_path = sys.argv[1]
read_names = set()
read_count = 0
with pysam.AlignmentFile(bam_path, "rb") as bamfile:
    for read in bamfile.fetch():
        read_count += 1
        read_names.add(read.query_name)

print(f"Total unique read names: {len(read_names)}")
print(f"Total reads: {read_count}")
