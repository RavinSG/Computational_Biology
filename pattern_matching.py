def trie_construction(patterns, end_flag=None):
    trie = dict()
    trie[0] = dict()

    for pattern in patterns:
        if end_flag is not None:
            pattern = pattern + end_flag
        parent_node = trie[0]
        for i in pattern:
            if i not in parent_node.keys():
                parent_node[i] = dict()

            parent_node = parent_node[i]

    return trie


def print_trie(trie, node, level):
    node_num = level + 1
    for child in trie[node]:
        print(f"{level}->{node_num}:{child}")
        node_num = print_trie(trie[node], child, node_num)

    return node_num
