import os



def _build_vocab(merges: list[str] = []):
    base_vocab_size = 256
    vocab = {k: bytes([k]) for k in range(base_vocab_size)}

    base_vocab_size+=1
    vocab[base_vocab_size] = '<|endoftext|>'.encode('utf-8')
    
    for m in merges:
        base_vocab_size+=1
        vocab[base_vocab_size] = m.encode('utf-8')

    return vocab

def _pre_tokenize(corpus: list[str]) -> list[str]:
    """
    Pre-tokenize the corpus into a list of pre-tokens based on whitespace.
    """
    tokens_list = [c.split() for c in corpus]
    tokens_list = [t for c in tokens_list for t in c]
    return tokens_list



def count_pre_tokens(pre_tokens):
    counter = {}
    for term in pre_tokens:
        if term in counter.keys():
            counter[term]+=1
        else:
            counter[term] = 1
    return counter

def transform_to_bytes(pre_token):
    """
    Transform a pre-token to a tuple of bytes.
    """
    pre_token_bytes = tuple(bytes([b]) for b in pre_token.encode('utf-8'))
    return pre_token_bytes


def get_pair_freq(freq_dict: dict):
    pairs_freq = {}
    for key, value in freq_dict.items():
        for first, second in zip(key, key[1:]):
            pair = (first, second)
            if pair in pairs_freq:
                pairs_freq[pair] += value
            else:
                pairs_freq[pair] = value
    return pairs_freq

def get_top_pair(stats):
    return max(stats, key=lambda p: (stats[p], p))

def merge_pair(freq_dict: dict, top_pair):
    new_freq_dict = {}
    for key, value in freq_dict.items():
        if top_pair in zip(key, key[1:]):
            i = 0
            new_key = []
            #print("found")
            while i < len(key):
                if key[i] == top_pair[0] and key[i+1] == top_pair[1]:
                    new_key.append(top_pair[0]+top_pair[1])
                    i += 2
                else:
                    new_key.append(key[i])
                    i += 1
            new_freq_dict[tuple(new_key)] = value
        else:
            new_freq_dict[key] = value
    return new_freq_dict

def tokenize(text, merges, vocab):
    pre_tokens = _pre_tokenize([text])
    pre_tokens_bytes = list(map(transform_to_bytes, pre_tokens))
    
    for merge in merges:
        new_pre_tokens_bytes = []
        for t in pre_tokens_bytes:
            i = 0
            new_t = []
            while i < len(t):
                if i < len(t)-1 and t[i]+t[i+1] == merge.encode('utf-8'):
                    new_t.append(merge.encode('utf-8'))
                    i += 2
                else:
                    new_t.append(t[i])
                    i += 1
            new_pre_tokens_bytes.append(tuple(new_t))
        pre_tokens_bytes = new_pre_tokens_bytes
    
    # decode
    tokens = []
    for byte_tuple in pre_tokens_bytes:
        for t in byte_tuple:
            tokens.append(t.decode('utf-8', errors='replace'))
    return tokens

def _load_texts_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines()]
    return texts


if __name__ == "__main__":

    # raw_texts = ["low low low low low",
    #       "lower lower widest widest widest",
    #       "newest newest newest newest newest newest"]
    raw_texts = _load_texts_from_file(os.path.join(os.path.dirname(__file__), "resources/raw_text.txt"))
    
    vocab = _build_vocab()
    print(f"Starting Vocab: {vocab}")

    pre_tokens = _pre_tokenize(raw_texts)
    print(f"\nPre-tokens: {pre_tokens}\n")

    pre_tokens_stats = count_pre_tokens(pre_tokens)
    print(f"\nPre-tokens stats: {pre_tokens_stats}\n")

    # transform to bytes
    pre_tokens_bytes = list(map(transform_to_bytes, pre_tokens))
 
    print(f"Pre-tokens bytes: {pre_tokens_bytes}\n")

    print("START TRAINING BPE\n")
    merges = []

    num_merges = 6
    for _ in range(num_merges):

        print(f"Merge iteration: {_+1}\n")
        # get the pair bytes frequency
        pair_stats = get_pair_freq(pre_tokens_stats)

        print(f"Pair stats: {pair_stats}\n")

        # get the top frequent pair
        top_pair = get_top_pair(pair_stats)

        print(f"Top pair: {top_pair}\n")


        # append the top pair to the merges
        merges.append(top_pair)

        # merge the pair
        pre_tokens_stats = merge_pair(pre_tokens_stats, top_pair)
        print(f"Updated pre-tokens stats: {pre_tokens_stats}\n")
    
    merges = [m[0]+m[1] for m in merges]
    print("merges:", merges)
    vocab = _build_vocab(merges)
    print("final vocab:", vocab)


    tokens = tokenize("newest", merges, vocab)
    print("tokens:", tokens)
