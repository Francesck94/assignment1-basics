from functools import lru_cache
import json


@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d


def _decode_term(byte_term: bytes, mapping_dict: dict[int, str]) -> str:
    """
    Decodes a byte term using the provided mapping dictionary.

    Args:
        byte_term (bytes): The byte sequence to decode.
        mapping_dict (dict[int, str]): The mapping dictionary from bytes to unicode characters.

    Returns:
        str: The decoded string representation of the byte term.
    """
    # get the corresponding integer
    term_mapped = [mapping_dict.get(idx) for idx in list(byte_term)]
    return "".join(term_mapped)


def _convert_to_byte(id_term: int) -> bytes:
    """
    Converts an integer ID to a single-byte bytes object.

    Args:
        id_term (int): The integer ID to convert.

    Returns:
        bytes: The corresponding single-byte bytes object.
    """
    return bytes([id_term])


def _encode_term(term: str, inv_mapping_dict: dict[str, int]) -> bytes:
    """
    Encodes a string term into a bytes object using the provided inverse mapping dictionary.

    Args:
        term (str): The string term to encode.
        inv_mapping_dict (dict[str, int]): The inverse mapping dictionary from unicode characters to bytes.

    Returns:
        bytes: The encoded bytes representation of the term.
    """
    mapped_list = [inv_mapping_dict.get(x) for x in term]
    encoded_list = [_convert_to_byte(x) for x in mapped_list]
    encoded_term = b"".join(encoded_list)
    return encoded_term


########## Vocab Serialization Utilities ##########

def save_vocab(vocab: dict[int, bytes], output_path: str):
    mapping_dict = gpt2_bytes_to_unicode()
    vocab_decoded = {index: _decode_term(v, mapping_dict) for index, v in vocab.items()}
    inv_vocab_decoded = {v: k for k, v in vocab_decoded.items()}
    with open(output_path, "w") as f:
        json.dump(inv_vocab_decoded, f)


def load_vocab(input_path: str) -> dict[int, bytes]:
    with open(input_path, "r") as f:
        inv_vocab_decoded = json.load(f)
    mapping_dict = gpt2_bytes_to_unicode()
    inv_mapping_dict = {v: k for k, v in mapping_dict.items()}
    vocab_decoded = {v: k for k, v in inv_vocab_decoded.items()}
    # vocab = {index: bytes([list(mapping_dict.keys())[list(mapping_dict.values()).index(term)] for term in decode_term(v.encode(), mapping_dict)]) for index, v in vocab_decoded.items()}

    vocab_loaded_encoded = {k: _encode_term(v, inv_mapping_dict) for k, v in vocab_decoded.items()}
    return vocab_loaded_encoded


########## Merges Serialization Utilities ##########

def save_merges(merges: list[tuple[bytes, bytes]], output_path: str):
    mapping_dict = gpt2_bytes_to_unicode()
    merges_decoded = [(_decode_term(a, mapping_dict), _decode_term(b, mapping_dict)) for a, b in merges]
    with open(output_path, "w") as f:
        for a, b in merges_decoded:
            f.write(f"{a} {b}\n")


def load_merges(input_path: str) -> list[tuple[bytes, bytes]]:
    mapping_dict = gpt2_bytes_to_unicode()
    inv_mapping_dict = {v: k for k, v in mapping_dict.items()}
    with open(input_path, "r") as f:
        merges_loaded = [tuple(line.strip().split()) for line in f]

    merges_loaded_encoded = [
        (_encode_term(a, inv_mapping_dict), _encode_term(b, inv_mapping_dict)) for a, b in merges_loaded
    ]
    return merges_loaded_encoded
