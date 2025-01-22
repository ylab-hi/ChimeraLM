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
