from utils import pre_tokenize_chunk, find_chunk_boundaries, TOKENIZE_PATTERN, split_on_special_tokens, get_pre_tokens_count, merge_pair

from multiprocessing import Pool
import regex as re

PARALLELIZE = True

SPECIAL_TOKENS = ["<|endoftext|>"]  # Add more special tokens as needed

filename = "./data/TinyStoriesV2-GPT4-valid.txt"

class Tokenizer:
    """
    Simple BPE Tokenizer.
    """
    def __init__(self, special_tokens: list[str]):
        self.base_vocab_size = 256
        self.merges = []
        self.special_tokens = special_tokens
        self.special_tokens_bytes = [s.encode('utf-8') for s in self.special_tokens]
        self._vocab = self._build_vocab()

        

    def encode(self, text: str):
        pass

    def decode(self, tokens: list[int]):
        pass
    

    def train(self, input_path: str, vocab_size:int, num_process:int=4, enable_mp:bool=True):
        """
        Function for training the tokenizer on the given text. It will learn merges and build the vocabulary.
        """
        with open(input_path, 'rb') as f:
            boundaries = find_chunk_boundaries(f, num_process, b"<|endoftext|>")

            # only for testing purposes, limit to the first 4 chunks
            # TODO: remove this line after testing
            boundaries = boundaries[:2]

        if enable_mp:
            pairs = list(zip(boundaries[:-1], boundaries[1:]))
            args = [(input_path, start, end, self.special_tokens) for start, end in pairs]

            with Pool(processes=num_process) as pool:
                    #results = pool.starmap(pre_tokenize_chunk, args)
                    chunks_count = pool.starmap(get_pre_tokens_count, args)

            # combine results in a single dictionary
            pre_tokens_count = {}
            for chunk in chunks_count:
                for term, count in chunk.items():
                    if term in pre_tokens_count:
                        pre_tokens_count[term] += count
                    else:
                        pre_tokens_count[term] = count

        #return pre_tokens_count
        pre_tokens_count_bytes = {self._convert_key_to_tuple_of_bytes(k): v for k, v in pre_tokens_count.items()}
        
        #return pre_tokens_count_bytes

        # get the frequency of pairs in the pre-token counts

        # Learn merges until we reach the desired vocab size
        num_merges_to_do = vocab_size - len(self._vocab) - len(self.special_tokens)

        pair_freq = {}
        top_pair = None
        while num_merges_to_do > 0:
            # get the frequency of each pair of consecutive bytes in the pre-token counts
            if top_pair is not None:
                pair_freq = self.get_pair_freq_update(pre_tokens_count_bytes, pair_freq, top_pair)
            else:
                pair_freq = self.get_pair_freq(pre_tokens_count_bytes)

            if not pair_freq:
                break  # No more pairs to merge

            # get the most frequent pair of consecutive bytes
            top_pair = self.get_top_pair(pair_freq)
            self.merges.append(top_pair)

            # merge the most frequent pair in the pre-token counts
            pre_tokens_count_bytes = merge_pair(pre_tokens_count_bytes, top_pair)

            # update the frequency of pairs after the merge
            # update the number of merges left to do
            num_merges_to_do -= 1

        # Build the final vocabulary with the learned merges
        self._vocab = self._build_vocab()
        return self._vocab, self.merges

    def _build_vocab(self):
        base_vocab_size = self.base_vocab_size
        vocab = {k: bytes([k]) for k in range(base_vocab_size)}
        #base_vocab_size = vocab_size - 1

        if self.merges:
            for m in self.merges:
                base_vocab_size+=1
                vocab[base_vocab_size] = b"".join(c for c in m) #m.encode('utf-8')

        if self.special_tokens:
            for token in self.special_tokens:
                base_vocab_size += 1
                vocab[base_vocab_size] = token.encode('utf-8')
        
        return vocab

    def get_pair_freq(self, freq_dict: dict):
        """ Calculate the frequency of each pair of consecutive bytes in the input dictionary.
        Args:
            freq_dict (dict): A dictionary where keys are tuples of bytes and values are their frequencies.

        Returns:
            dict: A dictionary with pairs of consecutive bytes as keys and their frequencies as values.
        """
        
        pairs_freq = {}
        for key, value in freq_dict.items():
            for first, second in zip(key, key[1:]):
                pair = (first, second)
                if pair in pairs_freq:
                    pairs_freq[pair] += value
                else:
                    pairs_freq[pair] = value
        return pairs_freq

    def get_pair_freq_update(self, freq_dict: dict, pairs_freq: dict, top_pair: tuple):
            """ Calculate the frequency of each pair of consecutive bytes in the input dictionary.
            Args:
                freq_dict (dict): A dictionary where keys are tuples of bytes and values are their frequencies.
    
            Returns:
                dict: A dictionary with pairs of consecutive bytes as keys and their frequencies as values.
            """
            

            #keys_to_update = [k for k in freq_dict.keys() if top_pair[0] in k and top_pair[1] in k]
            merged_pair = top_pair[0] + top_pair[1]
            # get the keys that contain the top pair

            pre_tokens_with_merged_pair = []
            for key in freq_dict.keys():
                for k in key:
                    if k == merged_pair:
                        pre_tokens_with_merged_pair.append(key)
                        break
                

            # for key, value in freq_dict.items():
            #     for first, second in zip(key, key[1:]):
            #         pair = (first, second)
            #         if pair in pairs_freq:
            #             pairs_freq[pair] += value
            #         else:
            #             pairs_freq[pair] = value

            for key in pre_tokens_with_merged_pair:
                value = freq_dict[key]
                
                for first, second in zip(key, key[1:]):
                    pair = (first, second)
                    
                    if pair in pairs_freq:
                        pairs_freq[pair] += value
                    else:
                        pairs_freq[pair] = value

            pairs_freq.pop(top_pair, None)
            return pairs_freq

    @staticmethod
    def get_top_pair(stats):
        return max(stats, key=lambda p: (stats[p], p))

    #temporary
    def _convert_key_to_tuple_of_bytes(self, key):
        """
        Convert a key to bytes.
        """
        return tuple(c.encode('utf-8') for c in key)

    @staticmethod
    def _load_corpus(input_path: str):
            with open(input_path, 'r') as f:
                corpus = f.read()
            return corpus

    
    
if __name__ == "__main__":
   import json
   from pathlib import Path
   import cProfile

   tokenizer = Tokenizer(special_tokens=SPECIAL_TOKENS)

   vocab_size = 1024
   vocab, merges = tokenizer.train(filename, vocab_size, num_process=4, enable_mp=PARALLELIZE)

   print_results = False

   if print_results:
       print(f"Vocab: {vocab}")
       print(f"Merges: {merges}")
   
   #print(f"Vocab: {vocab}")
   #print(f"Merges: {merges}")

   output_path = Path(__file__).parent
   output_path = output_path.joinpath('output_train')
   output_path.mkdir(parents=True, exist_ok=True)

   vocab_encoded = {k: v.decode('utf-8', errors="replace") for k, v in vocab.items()}
   with open(output_path.joinpath("vocab.json"), "w") as f:
       json.dump(vocab_encoded, f)

   with open(output_path.joinpath("merges.txt"), "w") as f:
       for merge in merges:
           print(merge)
           chars_decoded = [c.decode('utf-8', errors="replace") for c in merge]
           f.write(" ".join(chars_decoded) + "\n")

    