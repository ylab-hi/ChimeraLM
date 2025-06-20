import typer
import random

def read_sup(path):
    """Read the supptive file and return a dictionary of read name and support number."""
    result = {}
    with open(path) as f:
        for line in f:
            content = line.strip().split()
            result[content[0]] = int(content[1])
    return result
            
            
app = typer.Typer()

def is_postive(read_name: str, supptive_reads: dict[str, int]) -> bool:
    """Check if the read is positive data."""
    if read_name in supptive_reads:
        return supptive_reads[read_name] == 0
    else:
        return False


@app.command()
def select(supptive_file: str, output_file: str, total_data: int, trainning_ratio: float = 0.7, validation_ratio: float = 0.2, test_ratio: float = 0.1, positive_data_ratio: float = 0.6):
    supptive_reads = read_sup(supptive_file)

    group_by_support: dict[int, list[str]] = {}
    for read_name, support in supptive_reads.items():
        if support not in group_by_support:
            group_by_support[support] = []
        group_by_support[support].append(read_name)
    
    # use read with support number equal to 0 as positive data 
    positive_data = group_by_support[0]

    # use read with support number more than 1 as negative data
    negative_data = []
    for support, reads in group_by_support.items():
        if support > 1:
            negative_data.extend(reads)
    
    # print group by support
    for support, reads in group_by_support.items():
        print(f"support {support}: {len(reads)}")
    
    # shuffle the data
    random.shuffle(positive_data)
    random.shuffle(negative_data)

    # split the data include positive data and negative data
    train_positive_data = positive_data[:int(total_data * trainning_ratio * positive_data_ratio)]
    train_negative_data = negative_data[:int(total_data * trainning_ratio * (1 - positive_data_ratio))]
    validation_positive_data = positive_data[int(total_data * trainning_ratio * positive_data_ratio):int(total_data * (trainning_ratio + validation_ratio) * positive_data_ratio)]
    validation_negative_data = negative_data[int(total_data * trainning_ratio * (1 - positive_data_ratio)):int(total_data * (trainning_ratio + validation_ratio) * (1 - positive_data_ratio))]
    test_positive_data = positive_data[int(total_data * (trainning_ratio + validation_ratio) * positive_data_ratio):]
    test_negative_data = negative_data[int(total_data * (trainning_ratio + validation_ratio) * (1 - positive_data_ratio)):]

    train_data = train_positive_data + train_negative_data
    validation_data = validation_positive_data + validation_negative_data
    test_data = test_positive_data + test_negative_data

    print(f"total_data: {len(train_data) + len(validation_data) + len(test_data)}")
    print(f"selecting data {total_data}")
    print(f"train_data: {len(train_data)}: {len(train_positive_data)} positive, {len(train_negative_data)} negative")
    print(f"validation_data: {len(validation_data)}: {len(validation_positive_data)} positive, {len(validation_negative_data)} negative")
    print(f"test_data: {len(test_data)}: {len(test_positive_data)} positive, {len(test_negative_data)} negative")

    # shuffle the data
    random.shuffle(train_data)
    random.shuffle(validation_data)
    random.shuffle(test_data)

    #  write train data to file
    with open(f"{output_file}.{len(train_data)}.train.txt", "w") as f:
        for read_name in train_data:
            f.write(f"{read_name}\t{is_postive(read_name, supptive_reads)}\n")

    # write validation data to file
    with open(f"{output_file}.{len(validation_data)}.validation.txt", "w") as f:
        for read_name in validation_data:
            f.write(f"{read_name}\t{is_postive(read_name, supptive_reads)}\n")
    
    # write test data to file
    with open(f"{output_file}.{len(test_data)}.test.txt", "w") as f:
        for read_name in test_data:
            f.write(f"{read_name}\t{is_postive(read_name, supptive_reads)}\n")

if __name__ == "__main__":
    app()