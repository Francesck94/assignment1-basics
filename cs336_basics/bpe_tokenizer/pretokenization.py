import os
from typing import BinaryIO
import regex as re
from multiprocessing import Process
from multiprocessing import Pool
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#### pre tokenize utils #####

TOKENIZE_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

#TOKENIZE_PATTERN = re.compile(TOKENIZE_PATTERN)


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


def _split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:

    special_token_pattern = "|".join(re.escape(token) for token in special_tokens)
    split_pattern = re.compile(special_token_pattern)

    # Split the chunk on the special tokens
    indexes = re.finditer(split_pattern, text)
    sub_chunks = []
    last_index = 0

    for match in indexes:
        # print(match.group())
        # print(match.start(), match.end())

        sub_chunks.append(text[last_index : match.start()])
        sub_chunks.append(match.group())
        last_index = match.end()

    if last_index < len(text):
        sub_chunks.append(text[last_index:])
    if len(sub_chunks) == 0:
        sub_chunks = [text]
    return [sub_chunk for sub_chunk in sub_chunks if sub_chunk]


def pre_tokenize_text_in_chunks(text: str, special_tokens: list[str]):
    """
    Pre-tokenize a text in chunks.

    Args:
        text (str): The text chunk to pre-tokenize.
        special_tokens (list[str]): List of special tokens to split on.

    Returns:
        list[str]: List of pre-tokenized text chunks.
    """
    if len(special_tokens) > 0:
        chunk_without_special_tokens = _split_on_special_tokens(text, special_tokens)
    else:
        chunk_without_special_tokens = [text]

    special_token_pattern = "|".join(re.escape(token) for token in special_tokens)

    token_pattern_complete = TOKENIZE_PATTERN + f"|{special_token_pattern}"
    token_pattern_complete = re.compile(token_pattern_complete)
    # print(token_pattern_complete)
    text_chunks = []
    for sub_chunk in chunk_without_special_tokens:
        if sub_chunk in special_tokens:
            text_chunks.append(sub_chunk)
        else:
            text_chunks.extend([t.group() for t in re.finditer(token_pattern_complete, sub_chunk)])
        logger.debug("text_chunks after pre-tokenization: %s", text_chunks)
    return text_chunks