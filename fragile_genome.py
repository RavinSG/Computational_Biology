import numpy as np


def greedy_sort(perm):
    k = len(perm)
    perm = np.array(perm)
    steps = 0
    for i in range(k):
        if perm[i] != i + 1:
            steps += 1
            k_index = np.where(np.abs(perm) == i + 1)[0][0]
            perm = np.concatenate((perm[:i], (-1 * perm[i:k_index + 1][::-1]), perm[k_index + 1:]))
            print(" ".join(f'{x:+d}' for x in perm))
        if perm[i] != i + 1:
            steps += 1
            perm[i] = abs(i + 1)
            print(" ".join(f'{x:+d}' for x in perm))

    return perm, steps


def count_breakpoints(perm):
    k = len(perm)
    points = 0
    for i in range(k - 1):
        if perm[i + 1] - perm[i] != 1:
            points += 1
    if perm[0] != 1:
        points += 1
    if perm[-1] != k:
        points += 1

    return points
