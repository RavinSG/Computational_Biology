import numpy as np
from collections import defaultdict


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


def create_suffix_tree(text):
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


def longest_repeat(node, length, sub_string, shared=False):
    child_len = length
    max_str = sub_string
    if not shared:
        for child in node:
            if node[child] != -1:
                tree_len = longest_repeat(node[child], length + len(child), sub_string + child)
                if tree_len[0] > child_len:
                    child_len = tree_len[0]
                    max_str = tree_len[1]
    else:
        for child in node:
            if child != 'col' and node[child]['col'] == 2:
                tree_len = longest_repeat(node[child], length + len(child), sub_string + child, True)
                if tree_len[0] > child_len:
                    child_len = tree_len[0]
                    max_str = tree_len[1]

    return child_len, max_str


def colour_tree(node):
    child_cols = []
    for child in node:
        if node[child] == -1:
            if '#' in child:
                node[child] = {'col': 0, 'val': -1}
                child_cols.append(0)
            else:
                node[child] = {'col': 1, 'val': -1}
                child_cols.append(1)
        else:
            col = colour_tree(node[child])
            node[child]['col'] = col
            child_cols.append(col)

    if 2 in child_cols:
        return 2
    elif 0 in child_cols and 1 in child_cols:
        return 2
    elif 0 in child_cols:
        return 0
    else:
        return 1


def print_leaves(node):
    for child in node:
        if node[child] == -1:
            if '#' in child:
                print(child.split("#")[0])
        else:
            print_leaves(node[child])


def extract_substrings(root, node, length, sub_string, shared_strings):
    child_len = length
    max_str = sub_string
    end_point = None
    end_node = True
    for child in node:
        if child != 'col' and node[child]['col'] == 2:
            end_node = False
            tree_len = extract_substrings(root, node[child], length + len(child), sub_string + child, shared_strings)
            if node == root:
                shared_strings.append([tree_len[1], tree_len[-1]])
            if tree_len[0] > child_len:
                child_len = tree_len[0]
                max_str = tree_len[1]
                end_point = tree_len[-1]

    if end_node:
        return child_len, max_str, shared_strings, node
    else:
        return child_len, max_str, shared_strings, end_point


def search_suffix_tree(node, pattern, coloured=True):
    if coloured:
        if len(pattern) == 0:
            return True, node['col']
        for child in node:
            if child[0] == pattern[0] and child != 'col' and child != 'val':
                for i in range(min(len(child), len(pattern))):
                    if child[i] != pattern[i]:
                        return False, -1
                else:
                    return search_suffix_tree(node[child], pattern[len(child):], True)
        else:
            return False, -1


def find_shortest_non_shared(string_1, string_2):
    tree = create_suffix_tree(string_1 + "#" + string_2)
    root = tree[0]
    sub_strings = []

    colour_tree(root)
    extract_substrings(root, root, 0, "", sub_strings)

    lengths = []
    for i in sub_strings:
        lengths.append(len(i[0]))
    lengths, sub_strings = zip(*sorted(zip(lengths, sub_strings)))

    checked_strings = dict()
    min_len = float('inf')
    min_sub = ""

    for i in sub_strings:
        sub_string = i[0]
        for j in i[1].keys():
            if '#' in j[1:]:
                sub_string = sub_string + j[0]
                break

        node = tree[0]
        while len(sub_string) > 1:
            if sub_string in checked_strings:
                break
            tree_num = search_suffix_tree(node, sub_string, True)[1]
            if tree_num < 1:
                if min_len > len(sub_string):
                    min_len = len(sub_string)
                    min_sub = sub_string

            checked_strings[sub_string] = tree_num
            sub_string = sub_string[1:]

    return min_sub


def suffix_arrays(text):
    suffixes = [text[x:] for x in range(len(text))]
    indices = [x for x in range(len(text))]

    _, indices = zip(*sorted(zip(suffixes, indices)))

    return indices


def burrows_wheeler_transform(text):
    indices = suffix_arrays(text)
    transform = ""

    for i in indices:
        transform += text[i - 1]

    return transform


def create_mapping_matrix(first_col, last_col, reverse=False):
    char_count_first = defaultdict(int)
    char_count_last = defaultdict(int)
    mapping = defaultdict(str)

    for i, j in zip(first_col, last_col):
        char_first = f"{char_count_first[i]}_{i}"
        char_count_first[i] += 1

        char_last = f"{char_count_last[j]}_{j}"
        char_count_last[j] += 1

        if reverse:
            mapping[char_first] = char_last
        else:
            mapping[char_last] = char_first

    return mapping, char_count_first


def inverse_burrows_wheeler(last_col):
    last_col = list(last_col)
    first_col = last_col[::]
    first_col.sort()

    mapping, _ = create_mapping_matrix(first_col, last_col)

    string = ""
    node = '0_' + last_col[0]
    for _ in range(len(last_col)):
        string += mapping[node].split('_')[1]
        node = mapping[node]

    return string[1:] + "$"


def burrows_wheeler_matching(last_col, patterns):
    last_col = list(last_col)
    first_col = last_col[::]
    first_col.sort()

    match_count = []
    mapping, count = create_mapping_matrix(first_col, last_col, True)

    for pattern in patterns:
        pattern = pattern[::-1]
        prev_candidates = [f"{x}_{pattern[0]}" for x in range(count[pattern[0]])]
        for i in range(1, len(pattern)):
            next_candidates = []
            for candidate in prev_candidates:
                if mapping[candidate].split('_')[1] == pattern[i]:
                    next_candidates.append(mapping[candidate])

            prev_candidates = next_candidates[::]
        match_count.append(len(prev_candidates))

    return match_count


def last_to_first_mapping(last_column):
    last_column = list(last_column)
    indices = [x for x in range(len(last_column))]

    first_column, indices = zip(*sorted(zip(last_column, indices)))
    last_to_first = np.zeros(len(indices))

    for i in range(len(indices)):
        last_to_first[indices[i]] = i

    return first_column, last_to_first.astype(int)


def count_n(last_col, alphabet=None):
    if alphabet is None:
        alphabet = dict()
        letters = sorted(set(last_col))
        for i, letter in enumerate(letters):
            alphabet[letter] = i

    count = []
    line_count = [0] * len(alphabet)

    for char in last_col:
        count.append(line_count[::])
        line_count[alphabet[char]] += 1

    count.append(line_count[::])

    return np.array(count), alphabet


def fast_bw_matching(last_col, patterns):
    first_col, last_to_first = last_to_first_mapping(last_col)
    count_symbol, alphabet = count_n(last_col)

    pattern_count = []
    for pattern in patterns:
        pattern = pattern[::-1]
        top = 0
        bot = len(last_col) - 1
        arr_start = top
        while len(pattern) > 0:
            next_indices = count_symbol[top:bot + 2][:, alphabet[pattern[0]]]
            if next_indices[0] != next_indices[-1]:
                l_top = next_indices[0] + 1
                l_bot = next_indices[-1]
                idx = np.searchsorted(next_indices, [l_top, l_bot]) - 1

                top, bot = last_to_first[arr_start + idx[0]], last_to_first[arr_start + idx[1]]
                arr_start = top

                pattern = pattern[1:]
            else:
                pattern_count.append(0)
                break
        else:
            pattern_count.append(bot - top + 1)

    return pattern_count
