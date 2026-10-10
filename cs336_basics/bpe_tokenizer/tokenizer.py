from cs336_basics.bpe_tokenizer.utils import convert_key_to_tuple_of_bytes
import regex as re
from typing import Iterable, Iterator
import logging
import json
from cs336_basics.bpe_tokenizer.pretokenization import pre_tokenize_text_in_chunks, find_chunk_boundaries, TOKENIZE_PATTERN

#TOKENIZE_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




# def _split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:

#     special_token_pattern = "|".join(re.escape(token) for token in special_tokens)
#     split_pattern = re.compile(special_token_pattern)

#     # Split the chunk on the special tokens
#     indexes = re.finditer(split_pattern, text)
#     sub_chunks = []
#     last_index = 0

#     for match in indexes:
#         # print(match.group())
#         # print(match.start(), match.end())

#         sub_chunks.append(text[last_index : match.start()])
#         sub_chunks.append(match.group())
#         last_index = match.end()

#     if last_index < len(text):
#         sub_chunks.append(text[last_index:])
#     if len(sub_chunks) == 0:
#         sub_chunks = [text]
#     return [sub_chunk for sub_chunk in sub_chunks if sub_chunk]


# def _pre_tokenize_text_in_chunks(text: str, special_tokens: list[str]):
#     """
#     Pre-tokenize a text in chunks.

#     Args:
#         text (str): The text chunk to pre-tokenize.
#         special_tokens (list[str]): List of special tokens to split on.

#     Returns:
#         list[str]: List of pre-tokenized text chunks.
#     """
#     if len(special_tokens) > 0:
#         chunk_without_special_tokens = _split_on_special_tokens(text, special_tokens)
#     else:
#         chunk_without_special_tokens = [text]

#     special_token_pattern = "|".join(re.escape(token) for token in special_tokens)

#     token_pattern_complete = TOKENIZE_PATTERN + f"|{special_token_pattern}"
#     token_pattern_complete = re.compile(token_pattern_complete)
#     # print(token_pattern_complete)
#     text_chunks = []
#     for sub_chunk in chunk_without_special_tokens:
#         if sub_chunk in special_tokens:
#             text_chunks.append(sub_chunk)
#         else:
#             text_chunks.extend([t.group() for t in re.finditer(token_pattern_complete, sub_chunk)])
#         logger.debug("text_chunks after pre-tokenization: %s", text_chunks)
#     return text_chunks


def convert_merge_result(merge_str: str):
    """
    Convert a merge string result into a tuple of bytes.
    """
    res = merge_str.rstrip("\n")
    res = res.split(" ")
    if len(res) > 2:
        if res[0]:
            res = [res[0], " "]
        elif res[-1]:
            res = [" ", res[-1]]
        else:
            pass

    return tuple(bytes(c.encode("utf-8")) for c in res)


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
        # Load vocab
        with open(vocab_filepath, "r") as f:
            vocab = json.load(f)

        # vocab = {int(v): k.encode('utf-8') for k, v in vocab.items()}
        vocab = {int(v): Tokenizer.map_to_bytes(k) for k, v in vocab.items()}

        # Load merges
        with open(merges_filepath, "r") as f:
            merges = f.readlines()

        merges = list(map(convert_merge_result, merges))

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

            tokens_pair = self._get_pair_from_token(pre_tok_bytes)

            # if the token exists and has exactly one pair, proceed with merging
            if tokens_pair:
                # get the first pair to merge based on the merges dictionary
                pair_to_merge = min(tokens_pair, key=lambda x: self.merges_dict.get(x, float("inf")))

                while self.merges_dict.get(pair_to_merge) is not None:
                    # print("merging", pair_to_merge, "in", pre_tok_bytes)
                    pre_tok_bytes = self._merge_pair_in_token(pair_to_merge, pre_tok_bytes)
                    tokens_pair = self._get_pair_from_token(pre_tok_bytes)
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
    def convert_bytes_to_str(v: bytes) -> str:
        v_int = list(v)
        v_list = list(map(chr, v_int))
        v_dec = "".join(v_list)
        return v_dec

    @staticmethod
    def _merge_pair_in_token(pair, token):
        first, second = pair
        merged_token = []
        i = 0
        while i < len(token):
            if i < len(token) - 1 and token[i] == first and token[i + 1] == second:
                merged_token.append(first + second)
                i += 2
            else:
                merged_token.append(token[i])
                i += 1
        return tuple(merged_token)

    @staticmethod
    def _get_pair_from_token(token: tuple) -> list[tuple]:
        pair_list = []
        # for i in range(len(token) - 1):
        #    pair_list.append((token[i], token[i + 1]))
        for t1, t2 in zip(token, token[1:]):
            pair_list.append((t1, t2))
        return pair_list

    @staticmethod
    def _merge_pair_in_token(pair: tuple, token: tuple) -> tuple:
        first, second = pair
        merged_token = []
        i = 0
        while i < len(token):
            if i < len(token) - 1 and token[i] == first and token[i + 1] == second:
                merged_token.append(first + second)
                i += 2
            else:
                merged_token.append(token[i])
                i += 1
        return tuple(merged_token)

    @staticmethod
    def map_to_bytes(vocab_char):
        temp = list(map(ord, vocab_char))
        return bytes(temp)


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

    test_string = "Hello, how are you?"
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
