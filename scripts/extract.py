from pathlib import Path

import pyfastx
import typer

app = typer.Typer()

@app.command()
def extract(fq_file: str, train_file: str, validation_file: str, test_file: str):
    """Extract chimeric reads from a fastq file and rename read with target."""
    train_reads = {}
    validation_reads = {}
    test_reads = {}

    with open(train_file) as f:
        for line in f:
            content = line.strip().split()
            train_reads[content[0]] = content[1]

    with open(validation_file) as f:
        for line in f:
            content = line.strip().split()
            validation_reads[content[0]] = content[1]

    with open(test_file) as f:
        for line in f:
            content = line.strip().split()
            test_reads[content[0]] = content[1]


    train_fq = Path(train_file).with_suffix(".fastq").open("w")
    validation_fq = Path(validation_file).with_suffix(".fastq").open("w")
    test_fq = Path(test_file).with_suffix(".fastq").open("w")

    with pyfastx.Fastx(fq_file) as fq:
        for read_name, read_seq, read_qual in fq:
            if read_name in train_reads:
                target = 1 if train_reads[read_name] == "True" else 0
                train_fq.write(f"{read_name}|{target}\n{read_seq}\n+\n{read_qual}\n")
            elif read_name in validation_reads:
                target = 1 if validation_reads[read_name] == "True" else 0
                validation_fq.write(f"{read_name}|{target}\n{read_seq}\n+\n{read_qual}\n")
            elif read_name in test_reads:
                target = 1 if test_reads[read_name] == "True" else 0
                test_fq.write(f"{read_name}|{target}\n{read_seq}\n+\n{read_qual}\n")

    train_fq.close()
    validation_fq.close()
    test_fq.close()


if __name__ == "__main__":
    app()
