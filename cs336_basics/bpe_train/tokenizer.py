from cs336_basics.bpe_train.utils import find_chunk_boundaries, get_pre_tokens_count, convert_key_to_tuple_of_bytes
import copy
from multiprocessing import Pool
from collections import defaultdict
import regex as re

PARALLELIZE = True

SPECIAL_TOKENS = ["<|endoftext|>"]  # Add more special tokens as needed

filename = "./data/TinyStoriesV2-GPT4-valid.txt"
# filename = "./data/test_data.txt"
#filename = '/Users/a415137/personal_projects/cs336/assignment1-basics/tests/fixtures/corpus.en'
#filename = '/Users/a415137/personal_projects/cs336/assignment1-basics/tests/fixtures/tinystories_sample_5M.txt'


class Tokenizer:
    """
    Simple BPE Tokenizer.
    """

    def __init__(self, special_tokens: list[str]):
        self.base_vocab_size = 256
        self.merges = []
        self.special_tokens = special_tokens
        self.special_tokens_bytes = [s.encode("utf-8") for s in self.special_tokens]
        self._vocab = self._build_vocab()

    def encode(self, text: str):
        pass

    def decode(self, tokens: list[int]):
        pass

    def train(self, input_path: str, vocab_size: int, num_process: int = 4, enable_mp: bool = True):
        """
        Function for training the tokenizer on the given text. It will learn merges and build the vocabulary.
        """
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_process, b"<|endoftext|>")

            # only for testing purposes, limit to the first 4 chunks
            # TODO: remove this line after testing
            #boundaries = boundaries[:2]

        if enable_mp:
            pairs = list(zip(boundaries[:-1], boundaries[1:]))
            args = [(input_path, start, end, self.special_tokens) for start, end in pairs]

            with Pool(processes=num_process) as pool:
                # results = pool.starmap(pre_tokenize_chunk, args)
                chunks_count = pool.starmap(get_pre_tokens_count, args)

            # combine results in a single dictionary
            pre_tokens_count = {}
            for chunk in chunks_count:
                for term, count in chunk.items():
                    if term in pre_tokens_count:
                        pre_tokens_count[term] += count
                    else:
                        pre_tokens_count[term] = count

        # return pre_tokens_count as bytes tuples
        pre_tokens_count_bytes = {convert_key_to_tuple_of_bytes(k): v for k, v in pre_tokens_count.items()}

        # Learn merges until we reach the desired vocab size
        num_merges_to_do = vocab_size - self.base_vocab_size - len(self.special_tokens)

        pair_freq_dict, pair_pre_tokens_dict = self._get_pair_dict_and_freq(pre_tokens_count_bytes)

        # pair_freq_dict: key is a pair of consecutive bytes, value is the frequency of that pair
        # pair_pre_tokens_dict: key is a pair of consecutive bytes, value is the set of pre-tokens containing that pair

        top_pair = None
        while num_merges_to_do > 0:
            # get the most frequent pair of consecutive bytes e.g (b'a', b'b')
            top_pair = self._get_top_pair(pair_freq_dict)

            # append the most frequent pair to the list of merges
            self.merges.append(top_pair)


            # get the set of pre-tokens that contain the top pair
            # these pre-tokens will be updated to reflect the merge
            pre_tokens_to_change = pair_pre_tokens_dict.get(top_pair)

            # update the frequency dictionary and pre-token counts based on the merge
            pair_freq_dict, pre_tokens_count_bytes, pair_pre_tokens_dict = self.update_frequency_dict(
                pair_freq_dict, pre_tokens_count_bytes, pair_pre_tokens_dict, top_pair, pre_tokens_to_change
            )

            # remove pairs with zero frequency
            for pair in list(pair_freq_dict.keys()):
                if pair_freq_dict[pair] == 0:
                    del pair_freq_dict[pair]

            # for testing purposes, compare the pair frequency dictionary with a freshly computed one
            #test_pair_freq_dict, _ = self._get_pair_freq(pre_tokens_count_bytes)
            #assert test_pair_freq_dict == pair_freq_dict, "Mismatch in pair frequency dictionaries for merge {} after {} merges".format(top_pair, len(self.merges))

            # update the number of merges left to do
            num_merges_to_do -= 1

        # Build the final vocabulary with the learned merges
        self._vocab = self._build_vocab()
        return self._vocab, self.merges

    def _build_vocab(self):
        base_vocab_size = self.base_vocab_size
        vocab = {k: bytes([k]) for k in range(base_vocab_size)}
        # base_vocab_size = vocab_size - 1
        vocab_index = base_vocab_size - 1

        if self.merges:
            for m in self.merges:
                vocab_index += 1
                vocab[vocab_index] = b"".join(c for c in m)  # m.encode('utf-8')

        if self.special_tokens:
            for token in self.special_tokens:
                vocab_index += 1
                vocab[vocab_index] = token.encode("utf-8")

        return vocab

    #TODO: Consider removing self from other static methods if not needed
    @staticmethod
    def _get_pair_freq(freq_dict: dict):
        """Calculate the frequency of each pair of consecutive bytes in the input dictionary.
        Args:
            freq_dict (dict): A dictionary where keys are tuples of bytes and values are their frequencies.

        Returns:
            dict: A dictionary with pairs of consecutive bytes as keys and their frequencies as values.
        """
        pair_dict = {}
        index_pair = 0

        pairs_freq = {}
        for key, value in freq_dict.items():
            for first, second in zip(key, key[1:]):
                pair = (first, second)
                if pair in pairs_freq:
                    pairs_freq[pair] += value
                else:
                    pairs_freq[pair] = value
                    pair_dict[pair] = index_pair
                    index_pair += 1
        return pairs_freq, pair_dict

    @staticmethod
    def _get_pair_dict_and_freq(freq_dict: dict):
        """Calculate the frequency of each pair of consecutive bytes in the input dictionary.
        Args:
            freq_dict (dict): A dictionary where keys are tuples of bytes and values are their frequencies.

        Returns:
            tuple: A tuple containing two dictionaries:
                - A dictionary with pairs of consecutive bytes as keys and their frequencies as values.
                - A dictionary with pairs of consecutive bytes as keys and sets of pre-tokens containing the pair as values.
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

    def update_frequency_dict(
        self, pair_freq_dict, pre_tokens_count_bytes, pair_pre_tokens_dict, top_pair, pre_tokens_to_change):
        """
        Update the frequency dictionary and pre-token counts based on the merge of the top pair.

        Args:
            pair_freq_dict (dict): A dictionary with pairs of consecutive bytes as keys and their frequencies as values.
            pre_tokens_count_bytes (dict): A dictionary with pre-tokens as keys and their frequencies as values.
            pair_pre_tokens_dict (dict): A dictionary with pairs of consecutive bytes as keys and sets of pre-tokens containing the pair as values.
            top_pair (tuple): The most frequent pair of consecutive bytes to be merged.
            pre_tokens_to_change (list): A list of pre-tokens that need to be updated based on the merge.

        Returns:
            tuple: A tuple containing the updated dictionaries:
                - Updated pair frequency dictionary.
                - Updated pre-token count dictionary.
                - Updated pair-to-pre-tokens dictionary.
        """

        # Create copies of the input dictionaries
        #new_pre_tokens_count_bytes = copy.deepcopy(pre_tokens_count_bytes)
        new_pre_tokens_count_bytes = pre_tokens_count_bytes.copy()

        new_pair_freq_dict = pair_freq_dict.copy()
        #new_pair_freq_dict = copy.deepcopy(pair_freq_dict)

        new_pair_pre_tokens_dict = pair_pre_tokens_dict.copy()
        #new_pair_pre_tokens_dict = copy.deepcopy(pair_pre_tokens_dict)

        # loop over the pre_tokens that need to be changed
        for pre_token in pre_tokens_to_change:
            # merge the top pair in the current pre_tokens to get the new pre_token
            new_pre_token = self._merge_pair_in_token(top_pair, pre_token)

            # retrieve the number of occurrences for that pre_token in the original pre-token count dictionary
            old_count = pre_tokens_count_bytes.get(pre_token)

            # update the new frequency count for the new pre_token adding the old count since it represents the occurrences of the original pre_token
            if new_pre_token in new_pre_tokens_count_bytes:
                new_pre_tokens_count_bytes[new_pre_token] += old_count
            else:
                new_pre_tokens_count_bytes[new_pre_token] = old_count

            # get all the consecutive byte pair from the pre_token 
            # e.g. for pre_token b'hello', the consecutive byte pairs would be [(b'h', b'e'), (b'e', b'l'), (b'l', b'l'), (b'l', b'o')]
            old_pair_list = self._get_pair_from_token(pre_token)

            # loop over each pair in the old pre_token and decrement its frequency by the old count
            for pair in old_pair_list:
                # update the frequency of the old pair by subtracting the old count
                new_pair_freq_dict[pair] = new_pair_freq_dict.get(pair, 0) - old_count

                # retrieve the list of pre_tokens associated with this pair
                pre_tokens_list = list(new_pair_pre_tokens_dict.get(pair))
                # remove the old pre_token from the list if it exists
                if pre_tokens_list and pre_token in pre_tokens_list:
                    ## update the list of pre_tokens associated with this pair
                    pre_tokens_list.remove(pre_token)
                        
                # now the pair-to-pre_tokens mapping for this pair has been updated
                new_pair_pre_tokens_dict[pair] = set(pre_tokens_list)

            # update the pair frequency and pair-to-pre_tokens mapping for the new pre_token
            new_pair_list = self._get_pair_from_token(new_pre_token)
            for pair in new_pair_list:
                new_pair_freq_dict[pair] = new_pair_freq_dict.get(pair, 0) + old_count
                # add new pre tokens to the pair_pre_tokens_dict
                if pair not in new_pair_pre_tokens_dict:
                    new_pair_pre_tokens_dict[pair] = set()
                pre_tokens_list = list(new_pair_pre_tokens_dict.get(pair))
                if pre_tokens_list:
                    if new_pre_token not in pre_tokens_list:
                        pre_tokens_list.append(new_pre_token)
                else:
                    pre_tokens_list = [new_pre_token]
                new_pair_pre_tokens_dict[pair] = set(pre_tokens_list)
            del new_pre_tokens_count_bytes[pre_token]

        return new_pair_freq_dict, new_pre_tokens_count_bytes, new_pair_pre_tokens_dict

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

    #TODO: To remove, since it duplicates the functionality of _merge_pair_in_token
    @staticmethod
    def _merge_pair(freq_dict: dict, top_pair):
        """
        Merge the specified top pair in the frequency dictionary.

        Args:
            freq_dict (dict): A dictionary with keys as tuples of bytes and values as their frequencies.
            top_pair (tuple): The pair of consecutive bytes to merge.

        Returns:
            dict: A new frequency dictionary with the top pair merged.
        """
        new_freq_dict = {}
        for key, value in freq_dict.items():
            if top_pair in zip(key, key[1:]):
                i = 0
                new_key = []
                # print("found")
                while i < len(key):
                    if i == len(key) - 1:
                        new_key.append(key[i])
                        i += 1
                        continue
                    elif key[i] == top_pair[0] and key[i + 1] == top_pair[1]:
                        new_key.append(top_pair[0] + top_pair[1])
                        i += 2
                    else:
                        new_key.append(key[i])
                        i += 1
                new_freq_dict[tuple(new_key)] = value
            else:
                new_freq_dict[key] = value
        return new_freq_dict


    @staticmethod
    def _get_top_pair(stats):
        return max(stats, key=lambda p: (stats[p], p))

    @staticmethod
    def _get_pair_from_token(token):
        pair_list = []
        for i in range(len(token) - 1):
            pair_list.append((token[i], token[i + 1]))
        return pair_list

    @staticmethod
    def _load_corpus(input_path: str):
        with open(input_path, "r") as f:
            corpus = f.read()
        return corpus


if __name__ == "__main__":
    import json
    from pathlib import Path
    import cProfile

    tokenizer = Tokenizer(special_tokens=SPECIAL_TOKENS)

    vocab_size = 300
    vocab, merges = tokenizer.train(filename, vocab_size, num_process=4, enable_mp=PARALLELIZE)

    print_results = False

    if print_results:
        print(f"Vocab: {vocab}")
        print(f"Merges: {merges}")

    # print(f"Vocab: {vocab}")
    # print(f"Merges: {merges}")

    output_path = Path(__file__).parent
    output_path = output_path.joinpath("output_train")
    output_path.mkdir(parents=True, exist_ok=True)

    vocab_encoded = {k: v.decode("utf-8", errors="replace") for k, v in vocab.items()}
    with open(output_path.joinpath("vocab.json"), "w") as f:
        json.dump(vocab_encoded, f)

    with open(output_path.joinpath("merges.txt"), "w") as f:
        for merge in merges:
            #print(merge)
            chars_decoded = [c.decode("utf-8", errors="replace") for c in merge]
            f.write(" ".join(chars_decoded) + "\n")
