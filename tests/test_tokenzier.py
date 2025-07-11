import chimera

def test_character_tokenizer():
    sequence = "ATCG"
    tokenizer = chimera.data.tokenizer.CharacterTokenizer()

    encoded = tokenizer.encode(sequence)
    tokens = tokenizer.convert_ids_to_tokens(encoded)
    decoded = tokenizer.decode(encoded)

    expected_encoded = [0, 7, 10, 8, 9, 1]
    expected_tokens = ['[CLS]', 'A', 'T', 'C', 'G', '[SEP]']

    assert encoded == expected_encoded
    assert tokens == expected_tokens
    assert decoded == sequence

def test_character_tokenizer_with_truncation():
    sequence = "ATCG" * 10000
    tokenizer = chimera.data.tokenizer.CharacterTokenizer(model_max_length=30000)
    encoded = tokenizer.encode(sequence, truncation=True)
    print(len(encoded))

def test_kmer_tokenizer():
    sequence = "ATCGCGATCG"
    tokenizer = chimera.data.tokenizer.KmerTokenizer(k=3)

    encoded = tokenizer.encode(sequence)
    tokens = tokenizer.convert_ids_to_tokens(encoded)
    decoded = tokenizer.decode(encoded)

    expected_encoded = [0, 28, 114, 43, 64, 42, 61, 28, 114, 1]
    expected_tokens = ['[CLS]', 'ATC', 'TCG', 'CGC', 'GCG', 'CGA', 'GAT', 'ATC', 'TCG', '[SEP]']
    expected_decoded = "ATCGCGATCG"

    assert encoded == expected_encoded
    assert tokens == expected_tokens
    assert decoded == expected_decoded

def test_hyena_tokenizer():
    hyena_tokenizer = chimera.data.tokenizer.load_tokenizer_from_hyena_model("hyenadna-small-32k-seqlen")
    sequence = "ATCG" * 10000
    hyena_encoded = hyena_tokenizer.encode(sequence, truncation=True)
    print(len(hyena_encoded))
