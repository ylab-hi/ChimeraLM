import chimera
import pytest

@pytest.mark.slow
def test_data_module():
    # Load the training data
    test_data = "tests/data/tests.parquet"

    tokenizer = chimera.data.tokenizer.CharacterTokenizer(model_max_length=100000)
    fq_data_module  = chimera.data.fq.DataModule(tokenizer, test_data, batch_size=12)
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

    tokenizer = chimera.data.tokenizer.CharacterTokenizer(padding_side="left")
    fq_data_module  = chimera.data.fq.DataModule(tokenizer, test_data, batch_size=12)

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

    tokenizer = chimera.data.tokenizer.CharacterTokenizer(padding_side="right")
    fq_data_module  = chimera.data.fq.DataModule(tokenizer, test_data, batch_size=12)

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

    tokenizer = chimera.data.tokenizer.CharacterTokenizer(model_max_length=100, padding_side="left")
    fq_data_module  = chimera.data.fq.DataModule(tokenizer, test_data, batch_size=12)

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

    tokenizer = chimera.data.tokenizer.CharacterTokenizer(model_max_length=1000, padding_side="right")
    fq_data_module  = chimera.data.fq.DataModule(tokenizer, test_data, batch_size=12)

    fq_data_module.prepare_data()
    fq_data_module.setup()
    train_data_loader = fq_data_module.train_dataloader()
    data_iterator = iter(train_data_loader)

    assert data_iterator is not None

    d1 = next(data_iterator)
    print(d1)
