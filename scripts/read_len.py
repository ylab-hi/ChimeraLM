# /// script
# dependencies = [
#   "typer",
#   "numpy",
#   "pyfastx",
#   "pysam",
# ]
# ///

import typer
import numpy as np
import pyfastx
from pathlib import Path
import pysam

app = typer.Typer()


# Read the length of the reads in the fq and save the read length to a npz file
@app.command()
def fq(file: Path, chimeric_read_file: Path | None = None):
    """Read the length of the reads in the file"""

    if chimeric_read_file is not None:
        with open(chimeric_read_file, "r") as f:
            chimeric_reads = {line.strip(): 1 for line in f}
            print(f"{chimeric_read_file} {len(chimeric_reads)} chimeric reads")
    else:
        chimeric_reads = {}

    read_lengths = []
    fastq = pyfastx.Fastx(file)
    for name, seq, qual in fastq:
        if chimeric_read_file is not None and name in chimeric_reads:
            read_lengths.append(len(seq))
        else:
            read_lengths.append(len(seq))

    print(f"{file} {len(read_lengths)} reads")
    output_file = f"{file.stem}_read_lengths.npz"
    np.savez(output_file, read_lengths=read_lengths)


def is_chimeric(read: pysam.AlignedSegment):
    """Check if the read is chimeric"""
    return (
        not read.is_unmapped
        and read.has_tag("SA")
        and not read.is_secondary
        and not read.is_supplementary
    )


@app.command()
def bam(file: Path, min_mapq: int = 10):
    """Read the length of the chimeric reads in the file"""

    read_lengths = []
    num_chimeric_alignments = []

    bam = pysam.AlignmentFile(file, "rb")
    total_chimeric_reads = 0
    total_reads = 0
    save_chimeric_reads = 0
    for read in bam:
        total_reads += 1
        if is_chimeric(read):
            total_chimeric_reads += 1
            if read.mapq >= min_mapq:
                save_chimeric_reads += 1

                sa_string = read.get_tag("SA")
                num_chimeric_alignment = (
                    len([sa for sa in sa_string.split(";") if sa]) + 1
                )

                read_lengths.append(read.query_length)
                num_chimeric_alignments.append(num_chimeric_alignment)

    bam.close()

    out_file = f"{file.stem}_chimeric_read_lengths_min_mapq_{min_mapq}.npz"
    np.savez(
        out_file,
        read_lengths=read_lengths,
        num_chimeric_alignments=num_chimeric_alignments,
    )

    print(f"{file} {total_reads} total reads")
    print(f"{file} {total_chimeric_reads} total chimeric reads")
    print(f"{file} {save_chimeric_reads} saved chimeric reads")

    print(
        f"{file} {save_chimeric_reads / total_chimeric_reads} saved chimeric reads / total chimeric reads"
    )
    print(
        f"{file} {total_chimeric_reads / total_reads} total chimeric reads / total reads"
    )


if __name__ == "__main__":
    app()
