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
from typing import Optional
import typer
from typer import Typer
import pysam
import os
import sys

app = Typer()


def is_chimeric(read: pysam.AlignedSegment) -> bool:
    """Check if a read is chimeric based on SA tag presence."""
    return not read.is_unmapped and read.has_tag("SA")


@app.command()
def generate_test_data(
    input_bam: Path,
    chimera_artifacts_file: Path,
    output_bam: Path,
    number_of_chimera_artifacts: int = typer.Option(100, "--chimera-artifacts", "-c", help="Number of chimeric artifact reads to include"),
    factor: int = 2,
    seed: Optional[int] = None,
):
    """Generate test data from a BAM file and a chimera artifacts file.

    Args:
        input_bam: Input BAM file path
        chimera_artifacts_file: Text file with one read name per line (chimeric artifacts)
        output_bam: Output BAM file path
        number_of_chimera_artifacts: Number of chimeric artifact read names to include
        factor: Multiplier for non-chimeric reads (default: 2x chimera artifacts)
        seed: Random seed for reproducibility (optional)
    
    The output BAM will contain:
    - N chimeric artifact read names (from chimera_artifacts_file)
    - N/factor chimeric non-artifact read names (chimeric but not in chimera_artifacts_file)
    - N*factor non-chimeric read names
    
    Important: Sampling is done by read name (query_name), not by individual alignments.
    If a read has multiple alignments (primary, secondary, supplementary), ALL alignments
    for that read name are kept together in the output.
    """
    # Validate inputs
    if not input_bam.exists():
        print(f"Error: Input BAM file not found: {input_bam}", file=sys.stderr)
        raise SystemExit(1)
    
    if not chimera_artifacts_file.exists():
        print(f"Error: Chimera artifacts file not found: {chimera_artifacts_file}", file=sys.stderr)
        raise SystemExit(1)
    
    if number_of_chimera_artifacts <= 0:
        print(f"Error: number_of_chimera_artifacts must be positive, got {number_of_chimera_artifacts}", file=sys.stderr)
        raise SystemExit(1)
    
    if factor <= 0:
        print(f"Error: factor must be positive, got {factor}", file=sys.stderr)
        raise SystemExit(1)
    
    # Set random seed for reproducibility
    if seed is not None:
        random.seed(seed)
    
    # Load chimera artifacts
    try:
        chimera_artifacts = set(
            line.strip() for line in chimera_artifacts_file.read_text().splitlines() 
            if line.strip()  # Skip empty lines
        )
        print(f"Loaded {len(chimera_artifacts)} chimera artifact read names")
    except Exception as e:
        print(f"Error reading chimera artifacts file: {e}", file=sys.stderr)
        raise SystemExit(1)

    number_chimera_artifact_reads = number_of_chimera_artifacts
    number_chimeric_reads_not_in_chimera_artifacts = number_of_chimera_artifacts // factor
    number_non_chimeric_reads = number_of_chimera_artifacts * factor

    print(f"Target composition:")
    print(f"  - {number_chimera_artifact_reads} chimeric artifact reads (by read name)")
    print(f"  - {number_chimeric_reads_not_in_chimera_artifacts} chimeric non-artifact reads (by read name)")
    print(f"  - {number_non_chimeric_reads} non-chimeric reads (by read name)")
    print(f"\nNote: Each read name may have multiple alignments (primary/secondary/supplementary).")
    print(f"      All alignments for a read name are kept together.\n")

    # Collect reads from different categories
    # Key: read_name (query_name), Value: list of all alignments for that read
    # This ensures primary and secondary/supplementary alignments stay together
    chimeric_artifact_reads: dict[str, list[pysam.AlignedSegment]] = defaultdict(list)
    chimeric_non_artifact_reads: dict[str, list[pysam.AlignedSegment]] = defaultdict(list)
    non_chimeric_reads: dict[str, list[pysam.AlignedSegment]] = defaultdict(list)

    print(f"Scanning input BAM file: {input_bam}")
    try:
        with pysam.AlignmentFile(input_bam.as_posix(), "rb") as bam_in:
            for read in bam_in.fetch(until_eof=True):
                # Skip reads without a query name
                query_name = read.query_name
                if query_name is None:
                    continue
                
                # Group all alignments (primary, secondary, supplementary) by read name
                if is_chimeric(read):
                    if query_name in chimera_artifacts:
                        chimeric_artifact_reads[query_name].append(read)
                    else:
                        chimeric_non_artifact_reads[query_name].append(read)
                else:
                    non_chimeric_reads[query_name].append(read)
    except Exception as e:
        print(f"Error reading input BAM file: {e}", file=sys.stderr)
        raise SystemExit(1)

    # Count total alignment records
    total_chimeric_artifact_alignments = sum(len(alns) for alns in chimeric_artifact_reads.values())
    total_chimeric_non_artifact_alignments = sum(len(alns) for alns in chimeric_non_artifact_reads.values())
    total_non_chimeric_alignments = sum(len(alns) for alns in non_chimeric_reads.values())

    print(f"Found:")
    print(f"  - {len(chimeric_artifact_reads)} unique chimeric artifact read names ({total_chimeric_artifact_alignments} total alignments)")
    print(f"  - {len(chimeric_non_artifact_reads)} unique chimeric non-artifact read names ({total_chimeric_non_artifact_alignments} total alignments)")
    print(f"  - {len(non_chimeric_reads)} unique non-chimeric read names ({total_non_chimeric_alignments} total alignments)")

    # Validate that we have enough reads in each category
    if len(chimeric_artifact_reads) < number_chimera_artifact_reads:
        print(f"Warning: Not enough chimeric artifact reads. Requested {number_chimera_artifact_reads}, found {len(chimeric_artifact_reads)}", file=sys.stderr)
        number_chimera_artifact_reads = len(chimeric_artifact_reads)
    
    if len(chimeric_non_artifact_reads) < number_chimeric_reads_not_in_chimera_artifacts:
        print(f"Warning: Not enough chimeric non-artifact reads. Requested {number_chimeric_reads_not_in_chimera_artifacts}, found {len(chimeric_non_artifact_reads)}", file=sys.stderr)
        number_chimeric_reads_not_in_chimera_artifacts = len(chimeric_non_artifact_reads)
    
    if len(non_chimeric_reads) < number_non_chimeric_reads:
        print(f"Warning: Not enough non-chimeric reads. Requested {number_non_chimeric_reads}, found {len(non_chimeric_reads)}", file=sys.stderr)
        number_non_chimeric_reads = len(non_chimeric_reads)

    # Sample the required number of reads from each category
    sampled_chimeric_artifacts = random.sample(
        list(chimeric_artifact_reads.keys()), number_chimera_artifact_reads
    )
    sampled_chimeric_non_artifacts = random.sample(
        list(chimeric_non_artifact_reads.keys()), number_chimeric_reads_not_in_chimera_artifacts
    )
    sampled_non_chimeric = random.sample(
        list(non_chimeric_reads.keys()), number_non_chimeric_reads
    )

    # Create output directory if it doesn't exist
    output_bam.parent.mkdir(parents=True, exist_ok=True)

    # Write the sampled reads to the output BAM file
    print(f"Writing reads to output BAM file...")
    total_alignments_written = 0
    try:
        with pysam.AlignmentFile(input_bam.as_posix(), "rb") as bam_in:
            with pysam.AlignmentFile(output_bam.as_posix(), "wb", template=bam_in) as bam_out:
                # Write all sampled reads (including all alignments per read name)
                for read_name in sampled_chimeric_artifacts:
                    for read in chimeric_artifact_reads[read_name]:
                        bam_out.write(read)
                        total_alignments_written += 1
                for read_name in sampled_chimeric_non_artifacts:
                    for read in chimeric_non_artifact_reads[read_name]:
                        bam_out.write(read)
                        total_alignments_written += 1
                for read_name in sampled_non_chimeric:
                    for read in non_chimeric_reads[read_name]:
                        bam_out.write(read)
                        total_alignments_written += 1
    except Exception as e:
        print(f"Error writing output BAM file: {e}", file=sys.stderr)
        raise SystemExit(1)
    
    # Sort and index the output BAM file using pysam
    print(f"Sorting and indexing BAM file...")
    try:
        sorted_bam = output_bam.with_suffix(".sorted.bam")
        pysam.sort("-o", sorted_bam.as_posix(), output_bam.as_posix())
        pysam.index(sorted_bam.as_posix())
        
        # Move sorted BAM to replace original
        os.replace(sorted_bam.as_posix(), output_bam.as_posix())
        
        # Move the index file as well
        sorted_index = Path(str(sorted_bam) + ".bai")
        output_index = Path(str(output_bam) + ".bai")
        if sorted_index.exists():
            os.replace(sorted_index.as_posix(), output_index.as_posix())
    except Exception as e:
        print(f"Error sorting/indexing BAM file: {e}", file=sys.stderr)
        raise SystemExit(1)

    # Calculate alignment counts for the final summary
    sampled_chimeric_artifact_alignments = sum(
        len(chimeric_artifact_reads[name]) for name in sampled_chimeric_artifacts
    )
    sampled_chimeric_non_artifact_alignments = sum(
        len(chimeric_non_artifact_reads[name]) for name in sampled_chimeric_non_artifacts
    )
    sampled_non_chimeric_alignments = sum(
        len(non_chimeric_reads[name]) for name in sampled_non_chimeric
    )

    print(f"\n{'='*60}")
    print(f"Successfully generated test data with:")
    print(f"  - {len(sampled_chimeric_artifacts)} chimeric artifact read names ({sampled_chimeric_artifact_alignments} alignments)")
    print(f"  - {len(sampled_chimeric_non_artifacts)} chimeric non-artifact read names ({sampled_chimeric_non_artifact_alignments} alignments)")
    print(f"  - {len(sampled_non_chimeric)} non-chimeric read names ({sampled_non_chimeric_alignments} alignments)")
    print(f"\nTotal: {len(sampled_chimeric_artifacts) + len(sampled_chimeric_non_artifacts) + len(sampled_non_chimeric)} unique read names")
    print(f"       {total_alignments_written} total alignment records written")
    print(f"\nOutput written to: {output_bam}")
    print(f"Index written to: {output_bam}.bai")
    print(f"{'='*60}")


if __name__ == "__main__":
    app()
