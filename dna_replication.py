import numpy as np
from collections import defaultdict
from itertools import combinations, product

from params import *


def pattern_matching(strand: str, pattern: str) -> list:
    """
    Given a strand and a pattern, returns the indices of the starting locations where the patter is present in the
    strand.

    :param strand: A sequence of nucleotides
    :param pattern: The pattern to be searched in the strand
    :return: A list of starting locations
    """
    positions = []
    start = 0
    while True:
        start = strand.find(pattern, start)
        if start != -1:
            positions.append(start)
            start += 1
        else:
            return positions


def pattern_count(strand: str, pattern: str) -> int:
    return len(pattern_matching(strand, pattern))


def compute_frequencies(strand: str, k: int) -> dict:
    freq = defaultdict(int)
    for i in range(len(strand) - k + 1):
        freq[strand[i:i + k]] = freq[strand[i:i + k]] + 1

    return freq


def most_frequent_kmer(strand: str, k: int) -> list:
    """
    Find the most common k-length sequences of nucleotides occurring in the strand.

    :param strand: A sequence of nucleotides
    :param k: Length of the k_mer
    :return: A list of k_mers
    """
    k_mers = []
    freq = compute_frequencies(strand, k)
    max_count = max(freq.values())

    for x, y in freq.items():
        if y == max_count:
            k_mers.append(x)

    return k_mers


def get_complement(strand: str, reverse=True) -> str:
    """
    Return the reverse complement of the DNA sequence

    :param strand: A sequence of nucleotides
    :param reverse: True if the complement should be reversed
    :return: The (reversed) complement of a DNA sequence
    """
    complement = ''.join(COMPLIMENTS[x] for x in strand)
    if reverse:
        complement = complement[::-1]

    return complement


def pattern_to_number(pattern: str) -> int:
    nums = np.flip(np.array(range(len(pattern))))
    values = np.array([STR_TO_NUM[x] for x in pattern])

    return sum(values * (4 ** nums))


def number_to_pattern(number: int, k: int) -> str:
    pattern = ''
    while number > 3:
        pattern = NUM_TO_STR[number % 4] + pattern
        number = number // 4
    pattern = NUM_TO_STR[number % 4] + pattern
    pattern = 'A' * (k - len(pattern)) + pattern

    return pattern


def skew_loc(strand: str, loc='min') -> int:
    """
    Finds the pivoting point of the G-C graph where the value stops decreasing and starts to increase.

    :param strand: A sequence of nucleotides
    :param loc: Which pivoting point is needed, ['min', 'max']
    """
    values = [0]

    for x, y in enumerate(strand):
        values.append(values[x] + GC_SCORE[y])

    values = np.array(values)
    if loc == 'min':
        return np.where(values == values.min())[0]
    else:
        return np.where(values == values.max())[0]


def hamming_distance(strand1: str, strand2: str) -> int:
    """
    Returns the hamming distance between the given two strands
    """
    distance = 0
    for i in range(len(strand1)):
        if strand1[i] != strand2[i]:
            distance += 1

    return distance


def approximate_pattern_count(strand: str, pattern: str, d: int) -> int:
    """
    Given a strand, a pattern, and a distance d, find the number of occurrences of the patter with a maximum hamming
    distance of d.

    :param strand: A sequence of nucleotides
    :param pattern: The pattern to be searched in the strand
    :param d: Maximum hamming distance
    :return: The number of occurrences
    """
    s_l = len(strand)
    p_l = len(pattern)
    count = 0
    for i in range(s_l - p_l + 1):
        splice = strand[i: i + p_l]
        if (hamming_distance(splice, pattern)) <= d:
            count += 1

    return count


def get_neighbors(k_mer: str, d: int) -> list:
    """
    Given a k_mer returns all the possible k_mers that are at most d distance from the original k_mer.

    :param k_mer: A sequence of nucleotides
    :param d: Maximum hamming distance
    :return: All possible k_mers with hamming distance d
    """
    mismatches = [k_mer]
    for dist in range(1, d + 1):
        for change_indices in combinations(range(len(k_mer)), dist):
            for substitutions in product(*[SWAP_BASES[k_mer[i]] for i in change_indices]):
                new_mismatch = list(k_mer)
                for idx, sub in zip(change_indices, substitutions):
                    new_mismatch[idx] = sub
                mismatches.append(''.join(new_mismatch))
    return mismatches


def strand_score(strand, k_mer):
    length = len(k_mer)
    scores = []
    for i in range(len(strand) - length + 1):
        scores.append(hamming_distance(strand[i:i + length], k_mer))

    return [min(scores), np.argmin(scores)]


def motif_enumeration(strands, k, d):
    patterns = []
    candidate = strands[0]
    for i in range(len(candidate) - k + 1):
        splice = candidate[i:i + k]
        neighbours = get_neighbors(splice, d)
        for neighbour in neighbours:
            for strand in strands[1:]:
                if strand_score(strand, neighbour)[0] > d:
                    break
            else:
                patterns.append(neighbour)

    return list(set(patterns))


def frequent_words_with_mismatch(strand: str, k: int, d: int) -> list:
    """
    Given a sequence find the k_mers that appear the most with at most d distance with the present k_mers in the
    sequence.

    :param strand: A sequence of nucleotides
    :param k: Length of the k_mer
    :param d: Maximum hamming distance
    :return: A list of the most occurring possible k_mer
    """
    possible_k_mers = defaultdict(int)
    for i in range(len(strand) - k):
        splice = strand[i: i + k]
        neighbours = get_neighbors(splice, d)
        for neighbour in neighbours:
            possible_k_mers[neighbour] += 1

    max_count = max(possible_k_mers.values())
    return [x for x, y in possible_k_mers.items() if y == max_count]


def find_clumps(genome: str, k: int, l: int, t: int):
    clumps = set()
    frequencies = compute_frequencies(genome[:l], k)
    for x, y in frequencies.items():
        if y >= t:
            clumps.add(x)

    for i in range(1, len(genome) - l + 1):
        first_pattern = genome[i - 1:i - 1 + k]
        last_pattern = genome[i + l - k:i + l]
        frequencies[first_pattern] -= 1
        frequencies[last_pattern] += 1

        if frequencies[last_pattern] >= t:
            clumps.add(last_pattern)

    return clumps


def hanoi_towers(n, start_peg, destination_peg):
    if n == 1:
        print(start_peg, destination_peg)
    else:
        transition_peg = 6 - start_peg - destination_peg
        hanoi_towers(n - 1, start_peg, transition_peg)
        print(start_peg, destination_peg)
        hanoi_towers(n - 1, transition_peg, destination_peg)
    return


def corr(a, b):
    score = 0
    for i in range(len(b)):
        if a[i:] == b[:-i]:
            score += 1 / (2 ** i)

    return score
