import os
from pathlib import Path
from cs336_basics.bpe_tokenizer.tokenizer import Tokenizer
import random

TINY_STORIES_PATH = Path(__file__).parents[2].joinpath("data", "TinyStoriesV2-GPT4-train.txt")
SPECIAL_TOKEN = "<|endoftext|>"
if __name__ == "__main__":
    print("Tiny Stories Path:", TINY_STORIES_PATH)
    with open(TINY_STORIES_PATH, "r") as f:
        txt = f.read()


    txt_chunks = txt.split(SPECIAL_TOKEN)
    num_chunks = len(txt_chunks)

    # select 10 random chunks
    selected_chunks = random.sample(txt_chunks, 10)
    #for i, chunk in enumerate(selected_chunks):
    #    print(f"Chunk {i+1}:\n{chunk}\n{'-'*40}")
    print(f"Total number of chunks: {num_chunks}")
    tokenizer = Tokenizer.fr()

    