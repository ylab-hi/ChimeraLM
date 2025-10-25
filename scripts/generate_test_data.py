# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pysam",
#     "typer",
# ]
# ///
from collections import defaultdict
import random
from pathlib import Path
from typer import Typer
import pysam
import os
app = Typer()


def is_chimeric(read: pysam.AlignedSegment) -> bool:
    return not read.is_unmapped and read.has_tag("SA")

@app.command()
def generate_test_data(
    input_bam: Path,
    chimera_artifacts_file: Path,
    output_bam: Path,
    number_of_chimera_artifacts: int = 1000,
    factor: int = 2,
):
    """Generate test data from a BAM file and a chimera file.

    A BAM file is given, and a chimera file is given.
    The chimera file is a text file with one read name per line.
    we also need to write non-chimeric reads, and chimeric read not in the chimera file.
    The output BAM file is a BAM file with only the reads in the chimera file.
    """

    chimera_artifacts = set(line.strip() for line in chimera_artifacts_file.read_text().splitlines())

    number_chimera_artifact_reads = number_of_chimera_artifacts 
    number_chimeric_reads_not_in_chimera_artifacts = number_of_chimera_artifacts // factor
    number_non_chimeric_reads = number_of_chimera_artifacts * factor

    # the output bam include number_chimera_artifact_reads + number_chimeric_reads_not_in_chimera_artifacts + number_non_chimeric_reads

    # Collect reads from different categories
    chimeric_artifact_reads: dict[str, list[pysam.AlignedSegment]] = defaultdict(list)
    chimeric_non_artifact_reads: dict[str, list[pysam.AlignedSegment]] = defaultdict(list)
    non_chimeric_reads: dict[str, list[pysam.AlignedSegment]] = defaultdict(list)

    with pysam.AlignmentFile(input_bam.as_posix(), "rb") as bam_in:
        for read in bam_in.fetch(until_eof=True):
            if is_chimeric(read):
                if read.query_name in chimera_artifacts:
                    chimeric_artifact_reads[read.query_name].append(read)
                else:
                    chimeric_non_artifact_reads[read.query_name].append(read)
            else:
                non_chimeric_reads[read.query_name].append(read)

    # Sample the required number of reads from each category
    sampled_chimeric_artifacts =random.sample(chimeric_artifact_reads.keys(), number_chimera_artifact_reads)
    sampled_chimeric_non_artifacts = random.sample(chimeric_non_artifact_reads.keys(), number_chimeric_reads_not_in_chimera_artifacts)
    sampled_non_chimeric = random.sample(non_chimeric_reads.keys(), number_non_chimeric_reads)

    # Write the sampled reads to the output BAM file
    with pysam.AlignmentFile(input_bam.as_posix(), "rb") as bam_in:
        with pysam.AlignmentFile(output_bam.as_posix(), "wb", template=bam_in) as bam_out:
            # Write all sampled reads
            for read_name in sampled_chimeric_artifacts:
                for read in chimeric_artifact_reads[read_name]:
                    bam_out.write(read)
            for read_name in sampled_chimeric_non_artifacts:
                for read in chimeric_non_artifact_reads[read_name]:
                    bam_out.write(read)
            for read_name in sampled_non_chimeric:
                for read in non_chimeric_reads[read_name]:
                    bam_out.write(read)
    
    # Sort and index the output BAM file using pysam
    sorted_bam = output_bam.with_suffix(".sorted.bam")
    pysam.sort("-o", sorted_bam.as_posix(), output_bam.as_posix())
    pysam.index(sorted_bam.as_posix())
    os.replace(sorted_bam.as_posix(), output_bam.as_posix())

    print(f"Generated test data with:")
    print(f"  - {len(sampled_chimeric_artifacts)} chimeric artifact reads")
    print(f"  - {len(sampled_chimeric_non_artifacts)} chimeric non-artifact reads")
    print(f"  - {len(sampled_non_chimeric)} non-chimeric reads")
    print(f"Total: {len(sampled_chimeric_artifacts) + len(sampled_chimeric_non_artifacts) + len(sampled_non_chimeric)} reads")
    print(f"Output written to: {output_bam}")


if __name__ == "__main__":
    app()