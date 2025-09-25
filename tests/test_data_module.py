import chimeralm
import pytest

@pytest.mark.slow
def test_data_module():
    # Load the training data
    test_data = "tests/data/tests.parquet"

    tokenizer = chimeralm.data.tokenizer.CharacterTokenizer(model_max_length=100000)
    fq_data_module  = chimeralm.data.fq.DataModule(tokenizer, test_data, batch_size=12)
    fq_data_module.prepare_data()
    fq_data_module.setup()
    train_data_loader = fq_data_module.train_dataloader()
    data_iterator = iter(train_data_loader)

    assert data_iterator is not None

    d1 = next(data_iterator)
    print(d1)

@pytest.mark.slow
def test_data_module_padding_left1():
    # Load the training data
    test_data = "tests/data/tests.parquet"

    tokenizer = chimeralm.data.tokenizer.CharacterTokenizer(padding_side="left")
    fq_data_module  = chimeralm.data.fq.DataModule(tokenizer, test_data, batch_size=12)

    fq_data_module.prepare_data()
    fq_data_module.setup()
    train_data_loader = fq_data_module.train_dataloader()
    data_iterator = iter(train_data_loader)

    assert data_iterator is not None
    d1 = next(data_iterator)
    print(d1)

@pytest.mark.slow
def test_data_module_padding_right1():
    # Load the training data
    test_data = "tests/data/tests.parquet"

    tokenizer = chimeralm.data.tokenizer.CharacterTokenizer(padding_side="right")
    fq_data_module  = chimeralm.data.fq.DataModule(tokenizer, test_data, batch_size=12)

    fq_data_module.prepare_data()
    fq_data_module.setup()
    train_data_loader = fq_data_module.train_dataloader()
    data_iterator = iter(train_data_loader)

    assert data_iterator is not None
    d1 = next(data_iterator)
    print(d1)

@pytest.mark.slow
def test_data_module_padding_left():
    # Load the training data
    test_data = "tests/data/tests.parquet"

    tokenizer = chimeralm.data.tokenizer.CharacterTokenizer(model_max_length=100, padding_side="left")
    fq_data_module  = chimeralm.data.fq.DataModule(tokenizer, test_data, batch_size=12)

    fq_data_module.prepare_data()
    fq_data_module.setup()
    train_data_loader = fq_data_module.train_dataloader()
    data_iterator = iter(train_data_loader)

    assert data_iterator is not None

    d1 = next(data_iterator)
    assert d1['input_ids'].shape == (12, 98)

    print(d1)


@pytest.mark.slow
def test_data_module_padding_right():
    # Load the training data
    test_data = "tests/data/tests.parquet"

    tokenizer = chimeralm.data.tokenizer.CharacterTokenizer(model_max_length=1000, padding_side="right")
    fq_data_module  = chimeralm.data.fq.DataModule(tokenizer, test_data, batch_size=12)

    fq_data_module.prepare_data()
    fq_data_module.setup()
    train_data_loader = fq_data_module.train_dataloader()
    data_iterator = iter(train_data_loader)

    assert data_iterator is not None

    d1 = next(data_iterator)
    print(d1)


@pytest.mark.slow
def test_hyena_data_module_padding():
    # Load the training data
    test_data = "tests/data/test_chimric_reads.bam"

    tokenizer = chimeralm.data.tokenizer.load_tokenizer_from_hyena_model("hyenadna-small-32k-seqlen")
    fq_data_module  = chimeralm.data.bam.BamDataModule(tokenizer, train_data_path=test_data,
        val_data_path=test_data,
        test_data_path=test_data,
      batch_size=12)

    fq_data_module.prepare_data()
    fq_data_module.setup()
    train_data_loader = fq_data_module.train_dataloader()
    data_iterator = iter(train_data_loader)

    assert data_iterator is not None

    d1 = next(data_iterator)
    print(d1)