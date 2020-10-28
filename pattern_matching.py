from pprint import pprint


def trie_construction(patterns, end_flag=''):
    trie = dict()
    trie[0] = dict()

    for pattern in patterns:
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


def prefix_trie_matching(text, trie):
    node = trie[0]
    for i in range(len(text)):
        symbol = text[i]
        if symbol in node:
            node = node[symbol]
        else:
            break
    if len(node) == 0:
        return True
    else:
        return False


def trie_match(text, trie):
    len_t = len(text)
    i = 0
    while i < len_t:
        if prefix_trie_matching(text[i:], trie):
            print(i)
        i += 1


def add_failure_edge(trie, node, string):
    for i in range(1, len(string)):
        suffix = string[i:]
        parent = trie[0]
        for char in suffix:
            if char in parent:
                parent = parent[char]
            else:
                break
        else:
            if len(node) == 0:
                node[-1] = [parent, len(string)]
            else:
                node[-1] = parent
            break
    else:
        if len(node) == 0:
            node[-1] = [trie[0], len(string)]
        else:
            node[-1] = trie[0]

    for child in node:
        if child != -1:
            add_failure_edge(trie, node[child], string + child)


def aho_corasick_algorithm(text, patterns, end_flag=""):
    trie = trie_construction(patterns, end_flag)
    root = trie[0]
    for child in root:
        add_failure_edge(trie, root[child], child)
    trie[0][-1] = trie[0]

    node = root
    i = 0
    while i < len(text):
        if text[i] in node:
            node = node[text[i]]
            i += 1
        elif len(node) == 1:
            node, depth = node[-1]
            print(i - depth)
        else:
            if node == trie[0]:
                i += 1
            node = node[-1]

    if len(node) == 1:
        print(i - node[-1][1])

    return trie


def compress_tree(tree, node, string):
    if len(node) == 0:
        return string

    if len(node) == 1:
        child = list(node)[0]
        return string + compress_tree(tree, node[child], child)

    # Just return the node value if the node is branching
    mapping = {}
    for child in node:
        mapping[child] = compress_tree(tree, node[child], child)

    for i in mapping:
        value = node.pop(i)
        key = mapping[i]
        if key[-1] == "$":
            value = -1
        else:
            for j in range(1, len(key)):
                value = value[key[j]]
        node[key] = value
    return string


def suffix_tree(text):
    text = text + "$"
    tree = {0: dict()}
    txt_len = len(text)
    for i in range(txt_len):
        node = tree[0]
        suffix = text[i:]
        for char in suffix:
            if char not in node:
                node[char] = dict()
                node = node[char]
            else:
                node = node[char]

    compress_tree(tree, tree[0], "")
    return tree
