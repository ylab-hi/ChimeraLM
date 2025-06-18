from dataclasses import dataclass

import pysam
import typer
from aligntools import Cigar
from rich import print

# 'chr1,3929928,+,2S892M24D2062S,60,57;chr1,3929454,-,14S1217M119D1725S,19,363;'

@dataclass
class AlignmentSegment:
    """Alignment segment."""
    chromosome: str
    reference_start: int
    reference_end: int
    strand: str
    read_name: str
    cigar: Cigar

    @classmethod
    def from_str(cls, sa_part: str, read_name: str) -> 'AlignmentSegment':
        """Create an alignment segment from a string."""
        # Split the sa tag
        sa_part_parts = sa_part.split(',')
        chromosome = sa_part_parts[0]
        reference_start = int(sa_part_parts[1])
        strand = sa_part_parts[2]
        cigar = Cigar.coerce(sa_part_parts[3])
        reference_end = cigar.ref_length + reference_start
        return cls(chromosome, reference_start, reference_end, strand, read_name, cigar)

    @classmethod
    def from_pysam(cls, alignment: pysam.AlignedSegment) -> 'AlignmentSegment':
        """Create an alignment segment from a pysam alignment."""
        if alignment.reference_name is None:
            msg = f"Reference name is None for {alignment}"
            raise ValueError(msg)
        if alignment.query_name is None:
            msg = f"Query name is None for {alignment}"
            raise ValueError(msg)
        if alignment.cigarstring is None:
            msg = f"Cigarstring is None for {alignment}"
            raise ValueError(msg)
        if alignment.reference_end is None:
            msg = f"Reference end is None for {alignment}"
            raise ValueError(msg)

        return cls(alignment.reference_name,
         alignment.reference_start,
         alignment.reference_end,
         '+' if not alignment.is_reverse else '-',
         alignment.query_name, Cigar.coerce(alignment.cigarstring))

    def __str__(self) -> str:
        """String representation of the alignment segment."""
        return f"{self.__class__.__name__}({self.chromosome}:{self.reference_start}-{self.reference_end} {self.strand} {self.read_name})"

    def __repr__(self) -> str:
        """Representation of the alignment segment."""
        return self.__str__()

def is_primary_chimeric(alignment: pysam.AlignedSegment) -> bool:
    """Check if an alignment is a primary chimeric alignment.

    Args:
        alignment (pysam.AlignedSegment): The alignment to check.

    Returns:
        bool: _description_
    """
    # Check if the alignment has sa tag
    return bool(alignment.has_tag('SA') and not (alignment.is_secondary or alignment.is_supplementary))

def get_primary_chimeric_alignments(bam_file: str) -> dict[str, list[AlignmentSegment]]:
    """Get all primary chimeric alignments from a BAM file.

    Args:
        bam_file (str): The path to the BAM file.
    """
    chimeric_reads = {}
    bamfile  = pysam.AlignmentFile(bam_file , 'rb')

    for alignment in bamfile.fetch():
        if is_primary_chimeric(alignment):
            read_name = alignment.query_name
            if read_name is None:
                msg = f"Read name is None for {alignment}"
                raise ValueError(msg)
            chimeric_reads[read_name] = chimeric_read_to_alignment_segments(alignment)
    return chimeric_reads

def sort_alignement_seg_from_one_read(alignment_segments: list[AlignmentSegment]) -> list[AlignmentSegment]:
    """Sort alignment segments from one read."""
    return sorted(alignment_segments, key=lambda x: (x.chromosome, x.reference_start))

def chimeric_read_to_alignment_segments(chimeric_read: pysam.AlignedSegment) -> list[AlignmentSegment]:
    """Convert a chimeric read to a list of alignment segments."""
    # Get the sa tag
    sa = chimeric_read.get_tag('SA')
    if sa is None or not isinstance(sa, str):
        msg = f"SA tag is None for {chimeric_read}"
        raise ValueError(msg)

    # Get the read name
    read_name = chimeric_read.query_name
    if read_name is None:
        msg = f"Read name is None for {chimeric_read}"
        raise ValueError(msg)

    alignment_segments = []
    alignment_segments.append(AlignmentSegment.from_pysam(chimeric_read))

    # Split the sa tag
    sa_parts = sa.split(';')
    for sa_part in sa_parts:
        if sa_part == '':
            continue
        alignment_segment = AlignmentSegment.from_str(sa_part, read_name)
        alignment_segments.append(alignment_segment)

    return sort_alignement_seg_from_one_read(alignment_segments)

def compare_alignment_segments(alignment_segment1: list[AlignmentSegment], alignment_segment2: list[AlignmentSegment], threshold: int) -> bool:
    """Compare two alignment segments."""
    if len(alignment_segment1) != len(alignment_segment2):
        return False

    for as1, as2 in zip(alignment_segment1, alignment_segment2, strict=False):
        if as1.chromosome != as2.chromosome:
            return False
        if as1.strand != as2.strand:
            return False
        if abs(as1.reference_start - as2.reference_start) > threshold:
            return False
        if abs(as1.reference_end - as2.reference_end) > threshold:
            return False

    return True


def cal_sup_chimeric_reads(chimeric_reads: dict[str, list[AlignmentSegment]], sup_chimeric_reads: dict[str, list[AlignmentSegment]], threshold: int):
    result = {}
    for read_name, alignment_segments in chimeric_reads.items():
        for sup_alignment_segment in sup_chimeric_reads.values():
            if compare_alignment_segments(alignment_segments, sup_alignment_segment, threshold):
                result[read_name] = True
                break
            result[read_name] = False

    with open('sup_chimeric_reads.txt', 'w') as f:
        for read_name, is_sup in result.items():
            f.write(f"{read_name}\t{is_sup}\n")

app = typer.Typer()
@app.command()
def compare_chimeric(bam_file: str, sup_bam_file: str, threshold: int = 1000):
    """Compare chimeric reads."""
    chimeric_reads = get_primary_chimeric_alignments(bam_file)
    print(f"Found {len(chimeric_reads)} chimeric reads in {bam_file}")
    sup_chimeric_reads = get_primary_chimeric_alignments(sup_bam_file)
    print(f"Found {len(sup_chimeric_reads)} chimeric reads in {sup_bam_file}")
    cal_sup_chimeric_reads(chimeric_reads, sup_chimeric_reads, threshold)

if __name__ == "__main__":
    app()
