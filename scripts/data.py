import math
import random
from pathlib import Path

import typer


def read_support_file(path: str) -> dict[str, int]:
    """Read the supportive file and return a dictionary of read name and support number."""
    result = {}
    with open(path) as f:
        for line in f:
            content = line.strip().split()
            result[content[0]] = int(content[1])
    return result


app = typer.Typer()


def is_positive(read_name: str, supportive_reads: dict[str, int]) -> bool:
    """Check if the read is positive data."""
    return supportive_reads.get(read_name) == 0


@app.command()
def select(
    supportive_file: str,
    output_dir: str,
    total_data: int,
    training_ratio: float = 0.7,
    validation_ratio: float = 0.2,
    test_ratio: float = 0.1,
    positive_data_ratio: float = 0.6,
):
    """Select data for training, validation, and testing."""
    if not math.isclose(training_ratio + validation_ratio + test_ratio, 1.0):
        raise ValueError("The sum of training, validation, and test ratios must be 1.0.")

    supportive_reads = read_support_file(supportive_file)

    group_by_support: dict[int, list[str]] = {}
    for read_name, support in supportive_reads.items():
        group_by_support.setdefault(support, []).append(read_name)

    # print group by support
    for support, _reads in group_by_support.items():
        pass

    # Use read with support number equal to 0 as positive data
    positive_data = group_by_support.get(0, [])

    # Use read with support number more than 1 as negative data
    negative_data = [read for support, reads in group_by_support.items() if support >= 1 for read in reads]

    # Shuffle the data
    random.shuffle(positive_data)
    random.shuffle(negative_data)

    # Calculate the number of positive and negative samples
    num_positive = int(total_data * positive_data_ratio)
    num_negative = total_data - num_positive

    # Check if data is enough
    if len(positive_data) < num_positive:
        msg = f"Not enough positive data: have {len(positive_data)}, need {num_positive}"
        raise ValueError(msg)
    if len(negative_data) < num_negative:
        msg = f"Not enough negative data: have {len(negative_data)}, need {num_negative}"
        raise ValueError(msg)

    # Take required number of samples
    positive_samples = positive_data[:num_positive]
    negative_samples = negative_data[:num_negative]

    # Split positive data
    p_train_end = int(num_positive * training_ratio)
    p_val_end = p_train_end + int(num_positive * validation_ratio)
    train_positive = positive_samples[:p_train_end]
    validation_positive = positive_samples[p_train_end:p_val_end]
    test_positive = positive_samples[p_val_end:]

    # Split negative data
    n_train_end = int(num_negative * training_ratio)
    n_val_end = n_train_end + int(num_negative * validation_ratio)
    train_negative = negative_samples[:n_train_end]
    validation_negative = negative_samples[n_train_end:n_val_end]
    test_negative = negative_samples[n_val_end:]

    # Combine and shuffle datasets
    train_data = train_positive + train_negative
    random.shuffle(train_data)
    validation_data = validation_positive + validation_negative
    random.shuffle(validation_data)
    test_data = test_positive + test_negative
    random.shuffle(test_data)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    def write_data(filename, data):
        with open(output_path / filename, "w") as f:
            for read_name in data:
                f.write(f"{read_name}\t{is_positive(read_name, supportive_reads)}\n")

    write_data("train.txt", train_data)
    write_data("validation.txt", validation_data)
    write_data("test.txt", test_data)


if __name__ == "__main__":
    app()
