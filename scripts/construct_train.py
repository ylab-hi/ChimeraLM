from typing import Generator
from dataclasses import dataclass
import pysam
from aligntools import Cigar

# 'chr1,3929928,+,2S892M24D2062S,60,57;chr1,3929454,-,14S1217M119D1725S,19,363;'

@dataclass
class AlignmentSegment:
    chromosome: str
    reference_start: int
    reference_end: int
    strand: str
    read_name: str
    cigar: Cigar

    @classmethod
    def from_str(cls, sa_part: str, read_name: str) -> 'AlignmentSegment':
        # Split the sa tag
        sa_part_parts = sa_part.split(',')
        chromosome = sa_part_parts[0]
        reference_start = int(sa_part_parts[1])
        strand = sa_part_parts[2]
        cigar = Cigar.coerce(sa_part_parts[3])
        reference_end = cigar.ref_length + reference_start
        return cls(chromosome, reference_start, reference_end, strand, read_name, cigar)
    

def is_primary_chimeric(alignment: pysam.AlignedSegment) -> bool:
    """
    Check if an alignment is a primary chimeric alignment.

    Args:
        alignment (pysam.AlignedSegment): The alignment to check.

    Returns:
        bool: _description_
    """
    # Check if the alignment has sa tag

    if alignment.has_tag('sa'):
        if not (alignment.is_secondary or alignment.is_supplementary):
            return True
    return False

def get_primary_chimeric_alignments(bam_file: str) -> dict[str, list[AlignmentSegment]]:
    """
    Get all primary chimeric alignments from a BAM file.

    Args:
        bam_file (str): The path to the BAM file.
    """
    chimeric_reads = {}
    bamfile  = pysam.AlignmentFile(bam_file , 'rb')

    for alignment in bamfile.fetch():
        if is_primary_chimeric(alignment):
            read_name = alignment.query_name
            if read_name is None:
                raise ValueError(f"Read name is None for {alignment}")
            chimeric_reads[read_name] = list(chimeric_read_to_alignment_segments(alignment))

    return chimeric_reads


def chimeric_read_to_alignment_segments(chimeric_read: pysam.AlignedSegment) -> Generator[AlignmentSegment, None, None]:
    # Get the sa tag
    sa = chimeric_read.get_tag('sa')
    if sa is None or not isinstance(sa, str):
        raise ValueError(f"SA tag is None for {chimeric_read}")

    # Get the read name
    read_name = chimeric_read.query_name
    if read_name is None:
        raise ValueError(f"Read name is None for {chimeric_read}")

    # Split the sa tag
    sa_parts = sa.split(';')
    for sa_part in sa_parts:
        alignment_segment = AlignmentSegment.from_str(sa_part, read_name)
        yield alignment_segment


def compare_alignment_segments(alignment_segment1: list[AlignmentSegment], alignment_segment2: list[AlignmentSegment]) -> bool:
    """ Compare two alignment segments. """
    if len(alignment_segment1) != len(alignment_segment2):
        return False
    
    if len(alignment_segment1) > 2 or len(alignment_segment2) > 2:
        return False
    
    return True

     

if __name__ == "__main__":
    pass