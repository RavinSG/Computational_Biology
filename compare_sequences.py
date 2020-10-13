import numpy as np
from collections import OrderedDict


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
        distances[0, i] = distances[0, i - 1] + right[0, i - 1]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            distances[i, j] = max((distances[i - 1, j] + down_matrix[i - 1, j]),
                                  (distances[i, j - 1] + right[i, j - 1]))

    print(distances[-1, -1])


def string_backtrack(string_1, string_2):
    l_1 = len(string_1) + 1
    l_2 = len(string_2) + 1

    max_values = np.zeros((l_1, l_2))
    backtrack = np.zeros((l_1, l_2))

    for i in range(1, l_1):
        for j in range(1, l_2):
            match = 0
            if string_1[i - 1] == string_2[j - 1]:
                match = 1

            top = max_values[i - 1, j]
            left = max_values[i, j - 1]
            diag = max_values[i - 1, j - 1]

            max_values[i, j] = max(top, left, diag + match)

            if max_values[i, j] == top:
                backtrack[i, j] = 0
            elif max_values[i, j] == left:
                backtrack[i, j] = 1
            else:
                backtrack[i, j] = 2

    return backtrack


def print_longest_common_sequence(string_1, string_2):
    backtrack = string_backtrack(string_1, string_2)
    i = len(string_1)
    j = len(string_2)
    string = ""
    while i > 0:
        if backtrack[i, j] == 0:
            i = i - 1
        elif backtrack[i, j] == 1:
            j = j - 1
        else:
            string = (string_1[i - 1]) + string
            i = i - 1
            j = j - 1

    return string
