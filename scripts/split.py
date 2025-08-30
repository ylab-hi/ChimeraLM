import random
from pathlib import Path

import pyfastx
import typer

app = typer.Typer(add_completion=False, context_settings={"help_option_names": ["-h", "--help"]})


@app.command()
def split(input_file: Path, select_num: int):
    with open(input_file) as f:
        lines = f.readlines()

    random.shuffle(lines)
    output_file = input_file.parent / f"{input_file.stem}_select_{select_num}.txt"

    with open(output_file, "w") as f:
        for line in lines[:select_num]:
            f.write(line)


@app.command()
def extract(input_fq: Path, select_file: Path):
    with open(select_file) as f:
        lines = f.readlines()
    select_reads = {line.strip() for line in lines}

    fq = pyfastx.Fastx(input_fq)

    output_fq = input_fq.parent / f"{input_fq.stem}_select_{len(select_reads)}.fastq"

    with open(output_fq, "w") as f:
        for read_name, sequence, quality in fq:
            if read_name in select_reads:
                f.write(f"@{read_name}\n{sequence}\n+\n{quality}\n")


@app.command()
def add_target(input_fq: Path, target: int = 0):
    fq = pyfastx.Fastx(input_fq)
    output_fq = input_fq.parent / f"{input_fq.stem}_target_{target}.fastq"
    with open(output_fq, "w") as f:
        for read_name, sequence, quality in fq:
            f.write(f"@{read_name}|{target}\n{sequence}\n+\n{quality}\n")


@app.command()
def make_train(input_fq: Path, train_ratio: float = 0.7, val_ratio: float = 0.2, test_ratio: float = 0.1):
    fq = pyfastx.Fastx(input_fq)
    fq_reads = [(read_name, sequence, quality) for read_name, sequence, quality in fq]
    random.shuffle(fq_reads)
    total_reads = len(fq_reads)

    train_num = int(total_reads * train_ratio)
    val_num = int(total_reads * val_ratio)
    total_reads - train_num - val_num

    train_fq = input_fq.parent / f"{input_fq.stem}_train.fastq"
    val_fq = input_fq.parent / f"{input_fq.stem}_val.fastq"
    test_fq = input_fq.parent / f"{input_fq.stem}_test.fastq"

    with open(train_fq, "w") as f, open(val_fq, "w") as f2, open(test_fq, "w") as f3:
        for i, (read_name, sequence, quality) in enumerate(fq_reads):
            if i < train_num:
                f.write(f"@{read_name}\n{sequence}\n+\n{quality}\n")
            elif i < train_num + val_num and i >= train_num:
                f2.write(f"@{read_name}\n{sequence}\n+\n{quality}\n")
            else:
                f3.write(f"@{read_name}\n{sequence}\n+\n{quality}\n")


if __name__ == "__main__":
    app()
