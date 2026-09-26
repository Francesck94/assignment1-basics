from cs336_basics.bpe_train.utils import convert_key_to_tuple_of_bytes, find_chunk_boundaries, split_on_special_tokens, TOKENIZE_PATTERN
import regex as re
from typing import Iterable, Iterator

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []

        self.merges_dict = {m: i for i, m in enumerate(merges)}

        self.inv_vocab = {v: k for k, v in vocab.items()}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        pass

    def encode(self, text: str) -> list[int]:
        pre_tokens = self.pre_tokenize_text_in_chunks(text, self.special_tokens)
        pre_tokens_bytes = [convert_key_to_tuple_of_bytes(pre_tok) for pre_tok in pre_tokens]

        tokens_encoded = []
        for pre_tok_bytes in pre_tokens_bytes:

            print("pre-tokenized bytes:", pre_tok_bytes)
            tokens_pair = self._get_pair_from_token(pre_tok_bytes)

            # get the first pair to merge based on the merges dictionary
            pair_to_merge = min(tokens_pair, key=lambda x: self.merges_dict.get(x, float('inf')))
            
            while self.merges_dict.get(pair_to_merge) is not None:
                print("merging", pair_to_merge, "in", pre_tok_bytes)
                pre_tok_bytes = self._merge_pair_in_token(pair_to_merge, pre_tok_bytes)
                tokens_pair = self._get_pair_from_token(pre_tok_bytes)
                if not tokens_pair:
                    break
                pair_to_merge = min(tokens_pair, key=lambda x: self.merges_dict.get(x, float('inf')))

            tokens_ids = [self.inv_vocab[token] for token in pre_tok_bytes]
            tokens_encoded.extend(tokens_ids)
        return tokens_encoded

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass

    def decode(self, token_ids: list[int]) -> str:
        pass

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
    def _split_on_special_tokens(chunk: str, special_tokens: list[str]) -> list[str]:
        """
        Split a text chunk on special tokens and return a list of sub-chunks.
        """
        # Create a regex pattern to match any of the special tokens
        special_token_pattern = "|".join(re.escape(token) for token in special_tokens)
        split_pattern = re.compile(special_token_pattern)

        # Split the chunk on the special tokens
        sub_chunks = split_pattern.split(chunk)

        # Filter out empty strings and return the list of sub-chunks
        return [sub_chunk for sub_chunk in sub_chunks if sub_chunk.strip()]

    @staticmethod
    def _pre_tokenize_text_in_chunks(text: str, special_tokens: list[str]):
        """
        Pre-tokenize a text in chunks.

        Args:
            text (str): The text chunk to pre-tokenize.
            special_tokens (list[str]): List of special tokens to split on.

        Returns:
            list[str]: List of pre-tokenized text chunks.
        """
        
        chunk_without_special_tokens = Tokenizer._split_on_special_tokens(text, special_tokens)
        text_chunks = []
        for sub_chunk in chunk_without_special_tokens:
            text_chunks.extend([t.group() for t in re.finditer(TOKENIZE_PATTERN, sub_chunk)])
        return text_chunks

    @staticmethod
    def _get_pair_from_token(token: tuple) -> list[tuple]:
        pair_list = []
        #for i in range(len(token) - 1):
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