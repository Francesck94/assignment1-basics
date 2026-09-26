import json
import os
from typing import BinaryIO
import regex as re
from multiprocessing import Process
from multiprocessing import Pool
from collections import defaultdict

#### pre tokenize utils #####

TOKENIZE_PATTERN=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
TOKENIZE_PATTERN = re.compile(TOKENIZE_PATTERN)


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def count_pre_tokens(pre_tokens):
    counter = {}
    for term in pre_tokens:
        if term in counter.keys():
            counter[term]+=1
        else:
            counter[term] = 1
    return counter

def pre_tokenize_chunk(file_path, start_b, end_b, special_tokens):
    """
    Pre-tokenize a chunk of a file.

    Args:
        file_path (str): Path to the file.
        start_b (int): Start byte of the chunk.
        end_b (int): End byte of the chunk.
        special_tokens (list[str]): List of special tokens to split on.

    Returns:
        int: Number of pre-tokens in the chunk.
    """
    with open(file_path, "rb") as file:
        file.seek(start_b)
        chunk = file.read(end_b - start_b).decode("utf-8", errors="ignore")
        chunk_without_special_tokens = split_on_special_tokens(chunk, special_tokens)
        text_chunks = []
        for sub_chunk in chunk_without_special_tokens:
            text_chunks.extend([t.group() for t in re.finditer(TOKENIZE_PATTERN, sub_chunk)])
        return len(list(text_chunks))

def get_pre_tokens_count(file_path, start_b, end_b, special_tokens):
    with open(file_path, "rb") as file:
        file.seek(start_b)
        chunk = file.read(end_b - start_b).decode("utf-8", errors="ignore")
        chunk_without_special_tokens = split_on_special_tokens(chunk, special_tokens)
        text_chunks = []
        for sub_chunk in chunk_without_special_tokens:
            text_chunks.extend([t.group() for t in re.finditer(TOKENIZE_PATTERN, sub_chunk)])
        return count_pre_tokens(text_chunks)

def split_on_special_tokens(chunk: str, special_tokens: list[str]) -> list[str]:
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

def pre_tokenize_text_in_chunks(text: str, special_tokens: list[str]):
    """
    Pre-tokenize a text in chunks.

    Args:
        text (str): The text chunk to pre-tokenize.
        special_tokens (list[str]): List of special tokens to split on.

    Returns:
        list[str]: List of pre-tokenized text chunks.
    """
    
    chunk_without_special_tokens = split_on_special_tokens(text, special_tokens)
    text_chunks = []
    for sub_chunk in chunk_without_special_tokens:
        text_chunks.extend([t.group() for t in re.finditer(TOKENIZE_PATTERN, sub_chunk)])
    return text_chunks

################
#temporary
# def convert_key_to_tuple_of_bytes(key):
#     """
#     Convert a key to bytes.
#     """
#     return tuple(c.encode('utf-8') for c in key)

def convert_key_to_tuple_of_bytes(txt: str) -> tuple[bytes, ...]:
    """
    Convert a string to bytes.
    """
    return tuple(bytes([b]) for b in txt.encode('utf-8'))


# def convert_to_bytes(pre_token):
#     """
#     Transform a pre-token to a tuple of bytes.
#     """
#     pre_token_bytes = tuple(bytes([b]) for b in pre_token.encode('utf-8'))
#     return pre_token_bytes


def save_vocab_and_merges(vocab: dict, merges: list[tuple[bytes, bytes]], output_path: str):

    vocab_encoded = {k: v.decode("utf-8", errors="replace") for k, v in vocab.items()}

    # reverse vocab
    vocab_decoded = {v: k for k, v in vocab_encoded.items()}
    with open(os.path.join(output_path, "vocab.json"), "w") as f:
        json.dump(vocab_decoded, f)

    with open(os.path.join(output_path, "merges.txt"), "w") as f:
        for merge in merges:
            chars_decoded = [c.decode("utf-8", errors="replace") for c in merge]
            f.write(" ".join(chars_decoded) + "\n")



