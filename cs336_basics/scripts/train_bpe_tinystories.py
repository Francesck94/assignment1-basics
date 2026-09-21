"""
Train a byte-level BPE tokenizer on the TinyStories dataset, using a maximum vocabulary size
of 10,000. Make sure to add the TinyStories <|endoftext|> special token to the vocabulary.
Serialize the resulting vocabulary and merges to disk for further inspection. How many hours
and memory did training take? What is the longest token in the vocabulary? Does it make sense?
"""

from cs336_basics.bpe_train.tokenizer import Tokenizer
from cs336_basics.bpe_train.utils import save_vocab_and_merges
from pathlib import Path
import os
import time
import psutil
import threading
import tracemalloc

def poll_memory(output_path, interval=60):
    process = psutil.Process(os.getpid())
    while True:
        memory_usage = process.memory_info().rss / (1024 ** 2)  # in MB
        #print(f"Memory usage: {memory_usage} MB")
        with open(os.path.join(output_path, "memory_usage.log"), "a") as f:
            f.write(f"Memory usage: {memory_usage} MB\n")
        time.sleep(interval)

if __name__ == "__main__":
    ###### PATH SETUP ##########
    # path of the current script
    file_path = os.path.abspath(__file__)
    cwd = os.path.dirname(file_path)
    cwd = Path(cwd)
    root = cwd.parent.parent
    print("Root path:", root)

    input_path = root.joinpath( "data", "TinyStoriesV2-GPT4-train.txt")
    output_path = root.joinpath("output", "train_bpe_tinystories")
    os.makedirs(output_path, exist_ok=True)
    print("Input path:", input_path)
    print(os.path.exists(input_path))
    print("Output path:", output_path)
    print(os.path.exists(output_path))


    ##### TOKENIZER SETUP ##########
    SPECIAL_TOKENS = ["<|endoftext|>"]
    VOCAB_SIZE = 10000

    
    tokenizer = Tokenizer(special_tokens=SPECIAL_TOKENS)

    input_path = "./data/TinyStoriesV2-GPT4-train.txt"

    # Start memory polling in a separate thread
    #memory_thread = threading.Thread(target=poll_memory, args=(output_path,), daemon=True)
    #memory_thread.start()

    tracemalloc.start()
    start_time = time.time()
    vocab, merges = tokenizer.train(input_path, vocab_size=VOCAB_SIZE, show_progress=True, num_process=4)
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Current memory usage: {current / (1024 ** 2)} MB; Peak memory usage: {peak / (1024 ** 2)} MB")

    elapsed_time = end_time - start_time
    elapsed_time_minutes = elapsed_time / 60
    print(f"Training took {elapsed_time_minutes} minutes ({elapsed_time} seconds).")

    save_vocab_and_merges(vocab, merges, output_path)

    # save peak memory usage to a file
    with open(os.path.join(output_path, "memory_usage.log"), "a") as f:
        f.write(f"Peak memory usage: {peak / (1024 ** 2)} MB\n")

