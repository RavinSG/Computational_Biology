import numpy as np
from collections import defaultdict

from dna_replication import hamming_distance


def trie_construction(patterns: list, end_flag='') -> dict:
    """
    Creates a trie using the list of patters. The end flag is appended to each pattern to find the end points which are
    not located in leaves.

    :param patterns: A list of strings
    :param end_flag: Special character appended to end of every string
    :return: The created trie
    """
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


def print_trie(trie: dict, node, level: int) -> int:
    node_num = level + 1
    for child in trie[node]:
        print(f"{level}->{node_num}:{child}")
        node_num = print_trie(trie[node], child, node_num)

    return node_num


def prefix_trie_matching(text: str, trie: dict) -> bool:
    """
    Checks a text against a trie to check whether it is present.

    :param text: A sequence of characters
    :param trie: Trie generated form the original string
    :return: Whether the text is present or not in the trie
    """
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


def trie_match(text: str, trie: dict):
    """
    Given a text and a trie created using a list of patterns, prints the starting locations of the patterns located in
    the text.
    """
    len_t = len(text)
    i = 0
    while i < len_t:
        if prefix_trie_matching(text[i:], trie):
            print(i)
        i += 1


def add_failure_edge(trie: dict, node: list, string: str):
    """
    Adds a failure edge to each node in order to remove redundant pattern matching. These extra internal links allow
    fast transitions between failed string matches (e.g. a search for cat in a trie that does not contain cat, but
    contains cart, and thus would fail at the node prefixed by ca), to other branches of the trie that share a common
    prefix.

    This allows the automaton to transition between string matches without the need for backtracking.

    :param trie: Trie generated form the original string
    :param node: Node the failure edge should be added to
    :param string: String represented by the nodes upto the current node
    """
    for i in range(1, len(string)):
        # Check whether a pattern exists in the tree that is a suffix of the pattern responsible for the current node
        suffix = string[i:]
        parent = trie[0]
        for char in suffix:
            if char in parent:
                parent = parent[char]
            else:
                break
        else:
            # The failure edge is noted by the -1
            if len(node) == 0:
                node[-1] = [parent, len(string)]
            else:
                node[-1] = parent
            break
    else:
        # If no suffix is found backtrack to the root node
        if len(node) == 0:
            node[-1] = [trie[0], len(string)]
        else:
            node[-1] = trie[0]

    for child in node:
        if child != -1:
            add_failure_edge(trie, node[child], string + child)


def aho_corasick_algorithm(text: str, patterns: list, end_flag="") -> list:
    """
    Uses the Aho-Corasick algorithm to match the patterns instead of the searching every suffix of the text. This
    reduces the runtime of the algorithm from,
                        O(|text| * |longest pattern|) to O(|text| + |longest pattern| + |num matches|)

    :param text: A sequence of characters
    :param patterns: A list of strings
    :param end_flag: Special character appended to end of every string
    :return: The starting locations of patterns in the text
    """
    trie = trie_construction(patterns, end_flag)
    root = trie[0]
    for child in root:
        add_failure_edge(trie, root[child], child)
    trie[0][-1] = trie[0]

    node = root
    i = 0
    locations = []
    while i < len(text):
        if text[i] in node:
            node = node[text[i]]
            i += 1
        elif len(node) == 1:
            node, depth = node[-1]
            locations.append(i - depth)
        else:
            if node == trie[0]:
                i += 1
            node = node[-1]

    if len(node) == 1:
        locations.append(i - node[-1][1])

    return locations


def compress_tree(trie: dict, node: dict, string: str) -> str:
    """
    Remove all non-branching nodes from the trie and compress consecutive nodes in a non-branching path to a single
    node.

    :param trie: Trie generated form the original string
    :param node: Current node to be processed
    :param string: The string of the current node
    :return: String that represents the path from the current node to the end of the non-branching path
    """
    if len(node) == 0:
        return string

    # If the node is one-in-one-out return the current string + path
    if len(node) == 1:
        child = list(node)[0]
        return string + compress_tree(trie, node[child], child)

    # Return the node value if the node is branching and replace the edges with the compressed branches
    mapping = {}
    for child in node:
        mapping[child] = compress_tree(trie, node[child], child)

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


def create_suffix_tree(text: str) -> dict:
    """
    A compressed trie created from all the suffixes in text + "special_character"
    """
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


def longest_repeat(node: dict, length: int, sub_string: str, shared=False) -> tuple:
    """
    Find the longest substring that occurs in the text responsible for the trie. The trie is traversed to find the
    longest path that doesn't end in a leaf node. If the longest common substring occurring in two strings is to be
    found, the colouring of the nodes are used.

    :param node: Current node
    :param length: Length of the path till the node
    :param sub_string: Pattern spelled by the nodes in the current path
    :param shared: If true searches for the longest repeat in two strings
    :return: Length of the path and the string
    """
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


def colour_tree(node: dict) -> int:
    """
    Colours the nodes in a trie in the following manner.
        * A node is colored blue or red if all leaves in its subtree are all blue or all red, respectively
        * A node is colored purple if its subtree contains both blue and red leaves
    """
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


def print_leaves(node: dict):
    """
    Print the leaves of the trie. If a leaf contains parts from two strings only the first string is printed.
    """
    for child in node:
        if node[child] == -1:
            if '#' in child:
                print(child.split("#")[0])
        else:
            print_leaves(node[child])


def extract_substrings(root: dict, node: dict, length: int, sub_string: str, shared_strings: list) -> tuple:
    """
    Extract all the substrings shared by two strings.

    :param root: Root node of the trie
    :param node: Current node
    :param length: Length of the current substring
    :param sub_string: Substring upto the current node
    :param shared_strings: All substrings found
    :return: A tuple containing the length of the next node, longest substring, list of shared string and the next node
    """
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


def search_suffix_tree(node: dict, pattern: str, coloured=True) -> tuple:
    """
    Searches the suffix to check whether the pattern exists. If the tree is coloured, the color of the final node will
    also be returned. The colour represents whether the pattern is found in both trees or if only in one, the respective
    text.

    :param node: Current node
    :param pattern: Pattern to be search
    :param coloured: If true the colour of the node will also be returned
    :return: Whether the pattern exists and the node
    """
    if coloured:
        if len(pattern) == 0:
            return True, node['col']
        for child in node:
            # Since the nodes are compressed all characters should match before processing the next node
            if child[0] == pattern[0] and child != 'col' and child != 'val':
                for i in range(min(len(child), len(pattern))):
                    if child[i] != pattern[i]:
                        return False, -1
                else:
                    return search_suffix_tree(node[child], pattern[len(child):], True)
        else:
            return False, -1


def find_shortest_non_shared(string_1: str, string_2: str) -> str:
    """
    Finds the shortest substring of string_1 that does not appear in string_2.

    For a given shortest-non-shared-substring, the prefix of the string should be a common substring of both trees.
    Therefore the shortest-non-shared-substring should be of length at max |shortest common substring| + 1.

    The list of common substrings is searched first and then for each substring obtain the string of len |substring| + 1
    For each suffix of these stings, check whether it exists in the second string. Iterate through the list and find the
    shortest suffix that doesn't exist in the second string.

    :param string_1: String 1
    :param string_2: String 2
    :return: Shortest substring in string 1 that doesn't exist in string 2
    """
    tree = create_suffix_tree(string_1 + "#" + string_2)
    root = tree[0]
    sub_strings = []

    colour_tree(root)
    extract_substrings(root, root, 0, "", sub_strings)

    lengths = []
    for i in sub_strings:
        lengths.append(len(i[0]))
    # Sort the substring according to length since shorter ones have a higher chance
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
            # Add checked substrings to a dictionary to prevent searching them again
            checked_strings[sub_string] = tree_num
            sub_string = sub_string[1:]

    return min_sub


def generate_suffix_array(text: str) -> list:
    """
    Sort all suffixes of text lexicographically, assuming that "$" comes first in the alphabet. The indices of the
    starting locations of the sorted suffixes is the suffix array.
    """
    suffixes = [text[x:] for x in range(len(text))]
    indices = [x for x in range(len(text))]

    _, indices = zip(*sorted(zip(suffixes, indices)))

    return indices


def burrows_wheeler_transform(text: str) -> tuple:
    """
    Returns the Burrows-Wheeler transformed text.
    """
    indices = generate_suffix_array(text)
    transform = ""

    for i in indices:
        transform += text[i - 1]

    return transform, indices


def create_mapping_matrix(first_col: list, last_col: list, reverse=False) -> tuple:
    """
    Returns a dictionary mapping the characters in the first column to the last column on the Burrows-Wheeler
    transformation.

    The characters are stored with the position they are located in the first string, in the format of,
    {0_a: 1_b, 1_a: 1_c, ... }

    :param first_col: Sorted sequence
    :param last_col: Burrows-Wheeler transformed text
    :param reverse: If true the mapping is reversed
    :return: A dictionary containing the mapping
    """
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


def inverse_burrows_wheeler(last_col: str) -> str:
    """
    Reproduce the original string of the Burrows-Wheeler transform.
    """
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


def burrows_wheeler_matching(last_col: str, patterns: list) -> list:
    """
    Finds the number of pattern occurrences in a string using the Burrows-Wheeler transform and the mapping
    dictionaries.

    :param last_col: Burrows-Wheeler transformation of text
    :param patterns: Pattern to be searched
    :return: A list containing the positions
    """
    last_col = list(last_col)
    first_col = last_col[::]
    first_col.sort()

    match_count = []
    mapping, count = create_mapping_matrix(first_col, last_col, True)

    for pattern in patterns:
        pattern = pattern[::-1]
        # Populate the starting candidates with all positions of the last letter of the pattern
        prev_candidates = [f"{x}_{pattern[0]}" for x in range(count[pattern[0]])]
        for i in range(1, len(pattern)):
            next_candidates = []
            for candidate in prev_candidates:
                if mapping[candidate].split('_')[1] == pattern[i]:
                    next_candidates.append(mapping[candidate])

            prev_candidates = next_candidates[::]
        match_count.append(len(prev_candidates))

    return match_count


def last_to_first_mapping(last_column: str) -> np.ndarray:
    """
    Produces an array that stores the following information. Given a symbol at position i in the last column, stores its
    position in first column.
    """
    last_column = list(last_column)
    indices = [x for x in range(len(last_column))]

    first_column, indices = zip(*sorted(zip(last_column, indices)))
    last_to_first = np.zeros(len(indices))

    for i in range(len(indices)):
        last_to_first[indices[i]] = i

    return last_to_first.astype(int)


def get_alphabet(last_col: str) -> dict:
    """
    Returns a mapping of the alphabet used in the text to an integer.
    """
    alphabet = dict()
    letters = sorted(set(last_col))
    for i, letter in enumerate(letters):
        alphabet[letter] = i

    return alphabet


def count_n(last_col: str, alphabet: dict = None, checkpoints: int = None) -> np.ndarray:
    """
    Generates a matrix that stores the number of occurrences of each symbol of the alphabet in the first i positions of
    the Burrows-Wheeler transform in the ith position of the array.
                                        $  a  b  m  n  p  s
    E.g: count_n("smnpbnnaaaaa$")[3] = [0, 0, 0, 1, 1, 0, 1]

    :param last_col: Burrows-Wheeler transformation of text
    :param alphabet: Alphabet of the text
    :param checkpoints: Only the elements in indices divisible by the number will be returned
    :return: A count matrix
    """
    if alphabet is None:
        alphabet = get_alphabet(last_col)

    count = []
    line_count = [0] * len(alphabet)

    for char in last_col:
        count.append(line_count[::])
        line_count[alphabet[char]] += 1

    count.append(line_count[::])

    if checkpoints is not None:
        return np.array(count[::checkpoints])
    else:
        return np.array(count)


def fast_bw_matching(last_col: str, patterns: list, suffix_array: np.ndarray = None) -> tuple:
    """
    Uses the count array to quickly find the top and bottom indices of the next iteration, instead of going through
    every element of the currently selected rows.

    The first column is replaced with the count array and the first occurrence array (Since I'm storing the elements in
    the dictionary with the location the first occurrence array is not needed). The top and bottom indices are
    calculated by,
                    top <- first_occurrence(symbol) + count(top, last_col)[symbol]
                    bottom <- first_occurrence(symbol) + count(bottom + 1, last_col)[symbol] − 1

    :param last_col: Burrows-Wheeler transformation of text
    :param patterns: Patterns to be searched
    :param suffix_array: If provided the locations of the patterns are returned
    :return: An array with the number of patterns and the starting locations of the patterns
    """
    last_to_first = last_to_first_mapping(last_col)
    alphabet = get_alphabet(last_col)
    count_symbol = count_n(last_col, alphabet)
    starting_indices = []
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
            if suffix_array is not None:
                for i in range(top, bot + 1):
                    starting_indices.append(suffix_array[i])

            pattern_count.append(bot - top + 1)

    return pattern_count, starting_indices


def partial_suffix_array(text: str, checkpoint: int) -> dict:
    """
    The full suffix array is generated and then only the elements of this array that are divisible by checkpoint, are
    stored along with their indices i.
    """
    suffix_array = generate_suffix_array(text)
    partial_array = dict()
    for i, suffix in enumerate(suffix_array):
        if suffix % checkpoint == 0:
            partial_array[i] = suffix

    return partial_array


def first_occurrence_array(last_col: str, alphabet: dict) -> list:
    """
    Returns an array containing the first occurrence of each symbol in the sorted BW transformation.
    """
    first_col = list(last_col)
    first_col.sort()

    first_occurrences = []
    for i in alphabet:
        first_occurrences.append(first_col.index(i))

    return first_occurrences


def count_from_checkpoint(last_col, checkpoint, checkpoint_val, steps, symbol):
    """
    If a position is not available in the partial count array, start from the nearest previous checkpoint and count upto
    the necessary location.
    """
    for i in range(checkpoint, checkpoint + steps):
        if last_col[i] == symbol:
            checkpoint_val += 1
    return checkpoint_val


def optimized_burrows_wheeler_matching(text, patterns, alphabet=None, checkpoint=100, mismatches=0):
    """
    An optimized version of the BW matching that only used about 1.5 x |text| memory with increased run time. The only
    artifacts stored are the text, partial suffix array, first occurrence array and the checkpoint count matrix.

    This algorithm supports finding patters that match with the string with d number of mismatches. For a given pattern
    with d mismatches with the string, if we break the pattern into d+1 chunks at least one of them should have a
    perfect match. This is called the seed. Then using seeds the rest of the pattern is search with up to d mismatches.

    :param text: Text to be searched
    :param patterns: Patterns to be matched
    :param alphabet: The alphabet of the text
    :param checkpoint: Checkpoint distance
    :param mismatches: Number of mismatches allowed
    :return: The starting locations of the patterns
    """
    text = text + "$"
    suffix_array = generate_suffix_array(text)
    last_col, _ = burrows_wheeler_transform(text)
    if alphabet is None:
        alphabet = get_alphabet(last_col)
    first_occurrences = first_occurrence_array(last_col, alphabet)
    checkpoint_arrays = count_n(last_col, alphabet=alphabet, checkpoints=checkpoint)

    def find_indices(sub_pattern):
        starting_indices = []
        sub_pattern = sub_pattern[::-1]
        top = 0
        bot = len(last_col) - 1
        while len(sub_pattern) > 0:
            char = sub_pattern[0]
            # Use the first occurrence array and the partial count matrix to find the top and bottom positions of the
            # match
            if top % checkpoint != 0:
                l_top = count_from_checkpoint(last_col, (top // checkpoint) * checkpoint,
                                              checkpoint_arrays[top // checkpoint][alphabet[char]], top % checkpoint,
                                              char)
            else:
                l_top = checkpoint_arrays[top // checkpoint][alphabet[char]]
            bot = bot + 1
            if bot % checkpoint != 0:
                l_bot = count_from_checkpoint(last_col, (bot // checkpoint) * checkpoint,
                                              checkpoint_arrays[bot // checkpoint][alphabet[char]], bot % checkpoint,
                                              char)
            else:
                l_bot = checkpoint_arrays[bot // checkpoint][alphabet[char]]

            first_char = first_occurrences[alphabet[char]]
            top = first_char + l_top
            bot = first_char + l_bot - 1
            sub_pattern = sub_pattern[1:]

            # If there are no matches bot will rise above top
            if bot < top:
                break
        else:
            for i in range(top, bot + 1):
                starting_indices.append(suffix_array[i])

        return starting_indices

    indices = []
    for pattern in patterns:
        if mismatches == 0:
            start_indices = find_indices(pattern)
            if start_indices:
                indices += start_indices
        else:
            # Calculate the breaking position of the pattern
            break_points = np.floor(np.linspace(0, len(pattern), mismatches + 2)).astype(int)
            checked_pos = dict()
            # If a seed is checked against one point, the result will be the same for all matches of the same pattern
            # that has to match the seed against that position
            candidate_pos = dict()
            for start, end in zip(break_points, break_points[1:]):
                sub_string = pattern[start:end]
                candidate_pos[(start, end - start)] = find_indices(sub_string)
                checked_pos[(start, end - start)] = dict()

            for pos, pos_indices in candidate_pos.items():
                for pos_index in pos_indices:
                    distance = 0
                    # If the seed is checked on that position once, skip it. Since two seed of the same pattern could
                    # be perfect matches
                    if pos_index in checked_pos[pos].keys():
                        continue
                    else:
                        checked_pos[pos][pos_index] = True
                        for sibling_pos in candidate_pos:
                            if sibling_pos == pos:
                                continue
                            else:
                                str_start = pos_index + sibling_pos[0] - pos[0]
                                if str_start in checked_pos[sibling_pos]:
                                    distance = np.inf
                                    break
                                # Once a seed is checked against a position add it to the dict to prevent duplicate
                                # searches
                                checked_pos[sibling_pos][str_start] = True
                                sibling = pattern[sibling_pos[0]:sibling_pos[0] + sibling_pos[1]]
                                string_loc = (str_start, str_start + sibling_pos[1])
                                distance += hamming_distance(text[string_loc[0]:string_loc[1]], sibling)

                    if distance <= mismatches:
                        indices.append(pos_index - pos[0])

    return indices
