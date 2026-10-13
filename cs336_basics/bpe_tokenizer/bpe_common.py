def merge_pair_in_token(pair, token):
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


def get_pair_from_token(token):
    pair_list = []
    # for i in range(len(token) - 1):
    #    pair_list.append((token[i], token[i + 1]))
    for t1, t2 in zip(token, token[1:]):
        pair_list.append((t1, t2))
    return pair_list
