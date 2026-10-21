from cs336_basics.bpe_tokenizer.utils import (
    convert_key_to_tuple_of_bytes,
    convert_merge_result,
    map_vocab_char_to_bytes,
)
import regex as re
from typing import Iterable, Iterator
import logging
import json
from cs336_basics.bpe_tokenizer.pretokenization import (
    pre_tokenize_text_in_chunks,
    find_chunk_boundaries,
    TOKENIZE_PATTERN,
)
from cs336_basics.bpe_tokenizer.bpe_common import merge_pair_in_token, get_pair_from_token
from cs336_basics.bpe_tokenizer.serialize_utils import load_merges, load_vocab

# TOKENIZE_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []

        if self.special_tokens:
            self.special_tokens = sorted(self.special_tokens, key=len, reverse=True)

        self.merges_dict = {m: i for i, m in enumerate(merges)}

        self.inv_vocab = {v: k for k, v in vocab.items()}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):

        # Load vocab from the given file path using the utility function
        vocab = load_vocab(vocab_filepath)

        # Load merges from the given file path using the utility function
        merges = load_merges(merges_filepath)

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Encode a given text into a list of token IDs.

        Args:
            text (str): The input text to encode.

        Returns:
            list[int]: List of token IDs corresponding to the input text.
        """
        logger.debug("text: %s", text)
        pre_tokens = pre_tokenize_text_in_chunks(text, self.special_tokens)
        logger.debug("pre-tokens: %s", pre_tokens)

        if not self.special_tokens:
            pre_tokens_bytes = [convert_key_to_tuple_of_bytes(pre_tok) for pre_tok in pre_tokens]
        else:
            pre_tokens_bytes = [
                (
                    convert_key_to_tuple_of_bytes(pre_tok)
                    if pre_tok not in self.special_tokens
                    else pre_tok.encode("utf-8")
                )
                for pre_tok in pre_tokens
            ]

        special_tokens_bytes = [tok.encode("utf-8") for tok in self.special_tokens]
        tokens_encoded = []
        for pre_tok_bytes in pre_tokens_bytes:
            logger.debug("pre-tokenized bytes: %s", pre_tok_bytes)

            if pre_tok_bytes in special_tokens_bytes:
                tokens_encoded.extend([self.inv_vocab[pre_tok_bytes]])
                continue

            tokens_pair = get_pair_from_token(pre_tok_bytes)

            # if the token exists and has exactly one pair, proceed with merging
            if tokens_pair:
                # get the first pair to merge based on the merges dictionary
                pair_to_merge = min(tokens_pair, key=lambda x: self.merges_dict.get(x, float("inf")))

                while self.merges_dict.get(pair_to_merge) is not None:
                    # print("merging", pair_to_merge, "in", pre_tok_bytes)
                    pre_tok_bytes = merge_pair_in_token(pair_to_merge, pre_tok_bytes)
                    tokens_pair = get_pair_from_token(pre_tok_bytes)
                    # TODO: check if the condition: "len(tokens_pair[0]) < 2" can be removed

                    # if there are no more pairs to merge, break the loop
                    if not tokens_pair:
                        break

                    # if there is only one pair left and its length is less than 2, break the loop
                    if len(tokens_pair) == 1:
                        if len(tokens_pair[0]) < 2:
                            break

                    # Other cases: select the next pair to merge based on the merges dictionary
                    pair_to_merge = min(tokens_pair, key=lambda x: self.merges_dict.get(x, float("inf")))

            tokens_ids = [self.inv_vocab[token] for token in pre_tok_bytes]
            tokens_encoded.extend(tokens_ids)

        return tokens_encoded

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids):
        # Tokenizer can decode a list of integers into a string

        original_str = b"".join([self.vocab[idx] for idx in ids])
        original_str = original_str.decode("utf-8", errors="replace")
        return original_str

    @staticmethod
    def map_vocab_char_to_bytes(vocab_char: str) -> bytes:
        temp = list(map(ord, vocab_char))
        return bytes(temp)

    @staticmethod
    def convert_bytes_to_str(v: bytes) -> str:
        v_int = list(v)
        v_list = list(map(chr, v_int))
        v_dec = "".join(v_list)
        return v_dec


if __name__ == "__main__":
    from pathlib import Path
    import tiktoken

    reference_tokenizer = tiktoken.get_encoding("gpt2")

    cwd = Path(__file__).parent
    vocab_path = cwd / "output_train/vocab.json"
    merges_path = cwd / "output_train/merges.txt"
    special_tokens = ["<|endoftext|>"]

    tokenizer = Tokenizer.from_files(
        vocab_filepath=vocab_path, merges_filepath=merges_path, special_tokens=special_tokens
    )

    test_string = "Hello, how are you?<|endoftext|>"
    encoded = tokenizer.encode(test_string)
    print("Encoded:", encoded)
    decoded = tokenizer.decode(encoded)
    print("Decoded:", decoded)

    tokenized_string = [tokenizer.decode([x]) for x in encoded]
    tokenized_string.count("<|endoftext|>")
    print("Count of <|endoftext|> in tokenized string:", tokenized_string.count("<|endoftext|>"))

    # Compare the encoding of the test string with the reference tokenizer
    # reference_encoded = reference_tokenizer.encode(test_string, allowed_special={"<|endoftext|>"})
    # print("Reference Encoded:", reference_encoded)

    # # Compare the decoded output with the original test string
    # reference_decoded = reference_tokenizer.decode(reference_encoded)
    # print("Reference Decoded:", reference_decoded)

    # print("Count of <|endoftext|> in reference encoded string:", reference_decoded.count("<|endoftext|>"))
