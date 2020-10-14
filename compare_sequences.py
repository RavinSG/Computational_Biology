import numpy as np
from collections import OrderedDict, defaultdict

from params import *


def min_num_coins(money, coins):
    max_coin = max(coins)
    cache = OrderedDict({0: 0})

    for m in range(1, money + 1):
        try:
            cache[m] = min([cache[m - c] for c in coins if c <= m]) + 1
            while len(cache) > max_coin:
                cache.popitem(False)

        except ValueError:
            cache[m] = np.inf

    return cache


def manhattan_tourist_problem(n, m, down_matrix, right_matrix):
    distances = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        distances[i, 0] = distances[i - 1, 0] + down_matrix[i - 1, 0]

    for i in range(1, m + 1):
        distances[0, i] = distances[0, i - 1] + right_matrix[0, i - 1]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            distances[i, j] = max((distances[i - 1, j] + down_matrix[i - 1, j]),
                                  (distances[i, j - 1] + right_matrix[i, j - 1]))

    print(distances[-1, -1])


def string_backtrack(string_1, string_2, score_matrix=None, indel_penalty=5, local_align=False):
    l_1 = len(string_1) + 1
    l_2 = len(string_2) + 1

    max_values = np.zeros((l_1, l_2))
    backtrack = np.zeros((l_1, l_2))
    base_value = -np.inf
    if local_align:
        base_value = 0
    print("base", base_value)
    if score_matrix is None:
        for i in range(1, l_1):
            for j in range(1, l_2):
                match = 0
                if string_1[i - 1] == string_2[j - 1]:
                    match = 1

                top = max_values[i - 1, j]
                left = max_values[i, j - 1]
                diag = max_values[i - 1, j - 1]

                max_values[i, j] = max(top, left, diag + match, base_value)

                if max_values[i, j] == top:
                    backtrack[i, j] = 0
                elif max_values[i, j] == left:
                    backtrack[i, j] = 1
                else:
                    backtrack[i, j] = 2
    else:
        for i in range(1, l_1):
            max_values[i, 0] = max_values[i - 1, 0] - indel_penalty

        for i in range(1, l_2):
            max_values[0, i] = max_values[0, i - 1] - indel_penalty

        for i in range(1, l_1):
            for j in range(1, l_2):

                top = max_values[i - 1, j] - 5
                left = max_values[i, j - 1] - 5
                diag = max_values[i - 1, j - 1] + score_matrix[string_1[i - 1]][string_2[j - 1]]

                max_values[i, j] = max(top, left, diag, base_value)

                if max_values[i, j] == top:
                    backtrack[i, j] = 0
                elif max_values[i, j] == left:
                    backtrack[i, j] = 1
                else:
                    backtrack[i, j] = 2

    return backtrack, max_values


def find_longest_common_sequence(string_1, string_2, score_matrix=None):
    backtrack, _ = string_backtrack(string_1, string_2, score_matrix)
    i = len(string_1)
    j = len(string_2)
    align_1 = ""
    align_2 = ""
    while i > 0:
        value = backtrack[i, j]
        if value == 0:
            align_1 = string_1[i - 1] + align_1
            align_2 = "-" + align_2
            i = i - 1
        elif value == 1:
            align_1 = "-" + align_1
            align_2 = string_2[j - 1] + align_2
            j = j - 1
        else:
            align_1 = string_1[i - 1] + align_1
            align_2 = string_2[j - 1] + align_2
            i = i - 1
            j = j - 1

    return align_1, align_2


def find_local_alignment(string_1, string_2, score_matrix):
    backtrack, max_values = string_backtrack(string_1, string_2, score_matrix, local_align=True)
    align_1 = ""
    align_2 = ""

    end_node = np.unravel_index(np.argmax(max_values), max_values.shape)
    i, j = end_node
    while i > 0:
        value = backtrack[i, j]
        if value == 0:
            align_1 = string_1[i - 1] + align_1
            align_2 = "-" + align_2
            i = i - 1
        elif value == 1:
            align_1 = "-" + align_1
            align_2 = string_2[j - 1] + align_2
            j = j - 1
        else:
            if max_values[i, j] == 0:
                print(max_values[end_node[0], end_node[1]])
                return align_1, align_2
            else:
                align_1 = string_1[i - 1] + align_1
                align_2 = string_2[j - 1] + align_2
                i = i - 1
                j = j - 1
    return align_1, align_2


def find_longest_path(node, parents, path_lengths, backtrack):
    if node in path_lengths:
        return path_lengths[node], path_lengths, backtrack
    else:
        length = -np.inf
        parent_node = None
        if node in parents:
            for parent in parents[node]:
                parent_len, _, _ = find_longest_path(parent[0], parents, path_lengths, backtrack)
                parent_len += + parent[1]
                if parent_len > length:
                    length = parent_len
                    parent_node = parent

        path_lengths[node] = length
        if parent_node is not None:
            backtrack[node] = parent_node[0]

    return path_lengths[node], path_lengths, backtrack


def calculate_global_score(string_1, string_2):
    score = 0
    for x, y in zip(string_1, string_2):
        if x == '-' or y == '-':
            score -= 5
        else:
            score += BLOSUM_MATRIX[x][y]

    print(score)
