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
    Split the chunk on special tokens and return a list of sub-chunks.
    """
    # Create a regex pattern to match any of the special tokens
    special_token_pattern = "|".join(re.escape(token) for token in special_tokens)
    split_pattern = re.compile(special_token_pattern)

    # Split the chunk on the special tokens
    sub_chunks = split_pattern.split(chunk)

    # Filter out empty strings and return the list of sub-chunks
    return [sub_chunk for sub_chunk in sub_chunks if sub_chunk.strip()]

################
#temporary
def convert_key_to_tuple_of_bytes(key):
    """
    Convert a key to bytes.
    """
    return tuple(c.encode('utf-8') for c in key)

def convert_key_to_tuple_of_bytes_v2(key):
    """
    Convert a key to bytes.
    """
    return tuple(bytes([b]) for b in key.encode('utf-8'))


def convert_to_bytes(pre_token):
    """
    Transform a pre-token to a tuple of bytes.
    """
    pre_token_bytes = tuple(bytes([b]) for b in pre_token.encode('utf-8'))
    return pre_token_bytes

def merge_pair(freq_dict: dict, top_pair):
    new_freq_dict = {}
    for key, value in freq_dict.items():
        if top_pair in zip(key, key[1:]):
            i = 0
            new_key = []
            #print("found")
            while i < len(key):
                if i == len(key) - 1:
                    new_key.append(key[i])
                    i += 1
                    continue
                elif key[i] == top_pair[0] and key[i+1] == top_pair[1]:
                    new_key.append(top_pair[0]+top_pair[1])
                    i += 2
                else:
                    new_key.append(key[i])
                    i += 1
            new_freq_dict[tuple(new_key)] = value
        else:
            new_freq_dict[key] = value
    return new_freq_dict


def get_pair_dict_and_freq(freq_dict: dict):
        """ Calculate the frequency of each pair of consecutive bytes in the input dictionary.
        Args:
            freq_dict (dict): A dictionary where keys are tuples of bytes and values are their frequencies.

        Returns:
            tuple: A tuple containing two dictionaries:
                - dict: A dictionary with pairs of consecutive bytes as keys and their frequencies as values.
                - dict: A dictionary with pairs of consecutive bytes as keys and sets of pre-tokens containing the pair as values.
        """
        pair_pre_tokens_dict = defaultdict(set)
        pairs_freq_dict = {}
        for key, value in freq_dict.items():
            for first, second in zip(key, key[1:]):
                pair = (first, second)
                if pair in pairs_freq_dict:
                    pairs_freq_dict[pair] += value
                else:
                    pairs_freq_dict[pair] = value

                pair_pre_tokens_dict[pair].add(key)
        return pairs_freq_dict, pair_pre_tokens_dict


def get_pair_from_token(token):
    pair_list = []
    for i in range(len(token) - 1):
        pair_list.append((token[i], token[i + 1]))
    return pair_list

