from utils import pre_tokenize_chunk, find_chunk_boundaries, TOKENIZE_PATTERN, split_on_special_tokens, get_pre_tokens_count, count_pre_tokens
from multiprocessing import Pool
import regex as re

PARALLELIZE = True

SPECIAL_TOKENS = [b"<|endoftext|>"]  # Add more special tokens as needed

filename = "./data/TinyStoriesV2-GPT4-valid.txt"
    
    
if __name__ == "__main__":
    with open(filename, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        special_tokens_str = [token.decode("utf-8") for token in SPECIAL_TOKENS]


        if PARALLELIZE:
            pairs = list(zip(boundaries[:-1], boundaries[1:]))
            args = [(filename, start, end, special_tokens_str) for start, end in pairs]

            with Pool(processes=num_processes) as pool:
                 #results = pool.starmap(pre_tokenize_chunk, args)
                 results = pool.starmap(get_pre_tokens_count, args)

            #print("num of pre-tokenized chunks:", sum(results))
            print("pre-tokens:", len(results))
            # combine results in a single dictionary
            counter = {}
            for result in results:
                for term, count in result.items():
                    if term in counter:
                        counter[term] += count
                    else:
                        counter[term] = count
            print("total pre-tokens:", sum(counter.values()))
        else:
            text_chunks = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                # remove special tokens
                special_tokens_str = [token.decode("utf-8") for token in SPECIAL_TOKENS]
                chunk_without_special_tokens = split_on_special_tokens(chunk, special_tokens_str)
                print(f"Number of sub-chunks after splitting on special tokens: {len(chunk_without_special_tokens)}")
                # Run pre-tokenization on your chunk and store the counts for each pre-token
                # split the text using the pattern
                for sub_chunk in chunk_without_special_tokens:
                    text_chunks.extend(re.finditer(TOKENIZE_PATTERN, sub_chunk))
                #text_chunks = re.finditer(TOKENIZE_PATTERN, chunk)

            print(f"chunk size in chars: {len(list(text_chunks))}")
                #print(len(list(text_chunks)))
                #break