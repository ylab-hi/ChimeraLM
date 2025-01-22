# ruff: noqa: S107

from functools import partial

import torch
from datasets import Dataset
from deepbiop import fq
from transformers import (
    DataCollatorWithPadding,
    PreTrainedTokenizer,
)

IGNORE_INDEX = -100

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}


def parse_target(name):
    """Parse the target name to get the name and the target."""
    number_of_item = 2
    content = name.split("|")
    if len(content) < number_of_item:
        msg = f"Invalid target name: {name}"
        raise ValueError(msg)

    rid, target = content
    return rid, int(target)


def encode_qual(qual, offset=33):
    """Encode the quality score."""
    return list(fq.encode_qual(qual, offset))


def tokenize_and_align_labels_and_quals(data, tokenizer, max_length, pad_qual=0):
    """Tokenize the input data and align the labels and qualities."""
    tokenized_inputs = tokenizer(data["seq"], max_length=max_length, truncation=True, padding=True)
    seq_len = len(data["seq"])
    if seq_len >= max_length:
        quals = torch.cat((data["qual"][: max_length - 1], torch.tensor([pad_qual])))
    else:
        quals = torch.cat((data["qual"], torch.tensor([pad_qual])))

    normalized_quals = torch.nn.functional.normalize(quals.float(), dim=0)
    rid, target = parse_target(data["id"])
    tokenized_inputs.update({"input_quals": normalized_quals, "labels": target})
    return tokenized_inputs


def tokenize_and_align_labels_and_quals_ids(
    data, tokenizer, max_length, pad_qual=0, pad_label=IGNORE_INDEX, max_id_length=256
):
    """Tokenize the input data and align the labels and qualities."""
    tokenized_inputs = tokenizer(data["seq"], max_length=max_length, truncation=True, padding=True)

    seq_len = len(data["seq"])
    truncation = seq_len >= max_length

    if truncation:
        quals = torch.cat((data["qual"][: max_length - 1], torch.tensor([pad_qual])))
    else:
        quals = torch.cat((data["qual"], torch.tensor([pad_qual])))

    normalized_quals = torch.nn.functional.normalize(quals.float(), dim=0)

    rid, target = parse_target(data["id"])
    id_len = len(data["id"])

    new_id = [id_len, int(truncation)] + [ord(char) for char in rid]
    new_id = new_id[:max_id_length] if len(new_id) > max_id_length else new_id + [0] * (max_id_length - len(new_id))

    tokenized_inputs.update({"input_quals": normalized_quals, "id": new_id, "labels": target})
    return tokenized_inputs


def tokenize_dataset(dataset, tokenizer, max_length):
    """Tokenizes the input dataset using the provided tokenizer and aligns labels and qualities.

    Args:
        dataset (Dataset): The input dataset to be tokenized.
        tokenizer (Tokenizer): The tokenizer to be used for tokenization.
        max_length (int): The maximum length of the tokenized sequences.

    Returns:
        Tokenized dataset with aligned labels and qualities.

    Raises:
        ValueError: If the dataset is empty or if the tokenizer is not provided.
        TypeError: If the dataset is not of type Dataset or if the tokenizer is not of type Tokenizer.
    """
    if not dataset:
        raise ValueError("Input dataset is empty")
    if not tokenizer:
        raise ValueError("Tokenizer is not provided")
    if not isinstance(dataset, Dataset):
        raise TypeError("Input dataset must be of type Dataset")

    return dataset.map(
        partial(tokenize_and_align_labels_and_quals, tokenizer=tokenizer, max_length=max_length)
    ).remove_columns(["id", "seq", "qual"])


class CharacterTokenizer(PreTrainedTokenizer):
    """Character tokenizer."""

    model_input_names = ["input_ids"]

    def __init__(
        self,
        model_max_length: int | None = None,
        bos_token="[BOS]",
        eos_token="[SEP]",
        sep_token="[SEP]",
        cls_token="[CLS]",
        pad_token="[PAD]",
        mask_token="[MASK]",
        unk_token="[UNK]",
        **kwargs,
    ):
        """Character tokenizer for Hugging Face transformers.

        Args:
            characters (Sequence[str]): List of desired characters. Any character which
                is not included in this list will be replaced by a special token called
                [UNK] with id=6. Following are list of all of the special tokens with
                their corresponding ids:
                    "[CLS]": 0
                    "[SEP]": 1
                    "[BOS]": 2
                    "[MASK]": 3
                    "[PAD]": 4
                    "[RESERVED]": 5
                    "[UNK]": 6
                an id (starting at 7) will be assigned to each character.
            model_max_length (int): Model maximum sequence length.
        """
        self.characters = ("A", "C", "G", "T", "N")
        self.model_max_length = model_max_length

        self._vocab_str_to_int = {
            "[CLS]": 0,
            "[SEP]": 1,
            "[BOS]": 2,
            "[MASK]": 3,
            "[PAD]": 4,
            "[RESERVED]": 5,
            "[UNK]": 6,
            **{ch: i + 7 for i, ch in enumerate(self.characters)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}
        add_prefix_space = kwargs.pop("add_prefix_space", False)
        padding_side = kwargs.pop("padding_side", "right")

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            add_prefix_space=add_prefix_space,
            model_max_length=model_max_length,
            padding_side=padding_side,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        """Returns the size of the vocabulary."""
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str) -> list[str]:
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens to a single string."""
        return "".join(tokens)

    def get_special_tokens_mask(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = None,
        *,
        already_has_special_tokens: bool = False,
    ) -> list[int]:
        """Retrieve sequence ids from a token list that corresponds to special tokens."""
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        result = ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is not None:
            result += ([0] * len(token_ids_1)) + [1]
        return result

    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None
    ) -> list[int]:
        """Build model inputs from a sequence or a pair of sequences for sequence classification tasks by concatenating."""
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        result = cls + token_ids_0 + sep
        if token_ids_1 is not None:
            result += token_ids_1 + sep
        return result

    def get_vocab(self) -> dict[str, int]:
        """Get the vocabulary."""
        return self._vocab_str_to_int

    def decode(self, token_ids, *, skip_special_tokens=True, **kwargs):
        """Decode ids back to sequence string."""
        if isinstance(token_ids, dict):
            token_ids = token_ids["input_ids"]

        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        if isinstance(token_ids, list) and isinstance(token_ids[0], list):
            token_ids = token_ids[0]  # Take first sequence if batch

        tokens = [self._convert_id_to_token(i) for i in token_ids]
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in self.all_special_tokens]

        return self.convert_tokens_to_string(tokens)


class KmerTokenizer(PreTrainedTokenizer):
    """K-mer based tokenizer for DNA sequences.

    A tokenizer that splits DNA sequences into overlapping k-mers and converts them into token IDs.
    Inherits from PreTrainedTokenizer to integrate with the Hugging Face transformers library.

    The tokenizer handles special tokens like [CLS], [SEP], [PAD], etc. and can encode/decode
    between DNA sequences and token IDs.

    Example:
        >>> tokenizer = KmerTokenizer(k=6, model_max_length=512)
        >>> sequence = "ACGTACGTACGT"  # Input DNA sequence
        >>> encoded = tokenizer(sequence)  # Returns dict with input_ids
        >>> decoded = tokenizer.decode(encoded)  # Recovers original sequence
    """

    model_input_names = ["input_ids"]

    def __init__(
        self,
        *,
        k: int = 6,
        model_max_length: int | None = None,
        bos_token="[BOS]",
        eos_token="[SEP]",
        sep_token="[SEP]",
        cls_token="[CLS]",
        pad_token="[PAD]",
        mask_token="[MASK]",
        unk_token="[UNK]",
        **kwargs,
    ):
        """K-mer tokenizer for Hugging Face transformers.

        Args:
            k (int): Length of k-mers (default: 6)
            model_max_length (int): Maximum sequence length
            **kwargs: Additional arguments passed to PreTrainedTokenizer
        """
        self.k = k
        self.model_max_length = model_max_length

        # Generate all possible k-mers
        nucleotides = ["A", "C", "G", "T", "N"]
        kmers = self._generate_kmers(nucleotides, k)

        # Create vocabulary mappings
        self._vocab_str_to_int = {
            "[CLS]": 0,
            "[SEP]": 1,
            "[BOS]": 2,
            "[MASK]": 3,
            "[PAD]": 4,
            "[RESERVED]": 5,
            "[UNK]": 6,
            **{kmer: i + 7 for i, kmer in enumerate(kmers)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}

        add_prefix_space = kwargs.pop("add_prefix_space", False)
        padding_side = kwargs.pop("padding_side", "right")

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            add_prefix_space=add_prefix_space,
            model_max_length=model_max_length,
            padding_side=padding_side,
            **kwargs,
        )

    def _generate_kmers(self, alphabet, k):
        """Generate all possible k-mers from given alphabet."""
        if k == 1:
            return alphabet
        kmers = []
        for letter in alphabet:
            for smaller_kmer in self._generate_kmers(alphabet, k - 1):
                kmers.append(letter + smaller_kmer)
        return sorted(kmers)

    @property
    def vocab_size(self) -> int:
        """Returns the size of vocabulary."""
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str) -> list[str]:
        """Convert text into k-mers."""
        # Generate overlapping k-mers
        kmers = [text[i : i + self.k] for i in range(len(text) - self.k + 1)]
        # Handle sequences shorter than k
        if not kmers:
            return [text + "N" * (self.k - len(text)) if len(text) < self.k else text[: self.k]]
        return kmers

    def _convert_token_to_id(self, token: str) -> int:
        """Convert token to id, using UNK token for unknown k-mers."""
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        """Convert id back to token."""
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        """Convert k-mers back to sequence, handling overlaps."""
        if not tokens:
            return ""
        sequence = tokens[0]
        for token in tokens[1:]:
            sequence += token[-1]
        return sequence

    def get_special_tokens_mask(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = None,
        *,
        already_has_special_tokens: bool = False,
    ) -> list[int]:
        """Retrieve sequence ids from a token list that corresponds to special tokens."""
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        result = ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is not None:
            result += ([0] * len(token_ids_1)) + [1]
        return result

    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None
    ) -> list[int]:
        """Build model inputs from a sequence or a pair of sequences for sequence classification tasks by concatenating."""
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        result = cls + token_ids_0 + sep
        if token_ids_1 is not None:
            result += token_ids_1 + sep
        return result

    def get_vocab(self) -> dict[str, int]:
        """Get the vocabulary."""
        return self._vocab_str_to_int

    def decode(self, token_ids, *, skip_special_tokens=True, **kwargs):
        """Decode ids back to sequence string."""
        if isinstance(token_ids, dict):
            token_ids = token_ids["input_ids"]

        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        if isinstance(token_ids, list) and isinstance(token_ids[0], list):
            token_ids = token_ids[0]  # Take first sequence if batch

        tokens = [self._convert_id_to_token(i) for i in token_ids]
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in self.all_special_tokens]

        return self.convert_tokens_to_string(tokens)


def pad_without_fast_tokenizer_warning(tokenizer, *pad_args, **pad_kwargs):
    """Pads without triggering the warning about how using the pad function is sub-optimal when using a fast tokenizer."""
    # To avoid errors when using Feature extractors
    if not hasattr(tokenizer, "deprecation_warnings"):
        return tokenizer.pad(*pad_args, **pad_kwargs)

    # Save the state of the warning, then disable it
    warning_state = tokenizer.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    try:
        padded = tokenizer.pad(*pad_args, **pad_kwargs)
    finally:
        # Restore the state of the warning.
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = warning_state

    return padded


class DataCollator(DataCollatorWithPadding):
    """Data collator for tokenized datasets."""

    def torch_call(self, features):
        """Collate the input features."""
        label_name = "label" if "label" in features[0] else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0] else None

        qual_name = "input_quals"
        qual_pad_token_id = 0
        input_quals = [feature[qual_name] for feature in features]

        id_name = "id"  # for predction dataset

        no_labels_features = [
            {k: v for k, v in feature.items() if k not in [qual_name, label_name, id_name]} for feature in features
        ]

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            no_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if labels is None:
            return batch

        sequence_length = batch["input_ids"].shape[1]
        padding_side = self.tokenizer.padding_side

        def to_list(tensor_or_iterable):
            if isinstance(tensor_or_iterable, torch.Tensor):
                return tensor_or_iterable.tolist()
            return list(tensor_or_iterable)

        if padding_side == "right":
            batch[qual_name] = [
                to_list(qual) + [qual_pad_token_id] * (sequence_length - len(qual)) for qual in input_quals
            ]
        else:
            batch[qual_name] = [
                [qual_pad_token_id] * (sequence_length - len(qual)) + to_list(qual) for qual in input_quals
            ]

        batch[qual_name] = torch.tensor(batch[qual_name], dtype=torch.float32)
        batch[label_name] = torch.tensor(labels, dtype=torch.int64)

        # for predction dataset and save id feature
        if id_name in features[0]:
            batch[id_name] = torch.tensor([to_list(feature[id_name]) for feature in features], dtype=torch.int8)

        return batch
