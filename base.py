import numpy as np
from itertools import combinations, product
from collections import defaultdict, Counter

NUCLEOTIDES = ['A', 'T', 'C', 'G']
COMPLIMENTS = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}


def count_nucleotides(strand: str) -> dict:
    """
    Returns a dictionary containing the number of times each nucleotide is present in the strand.

    :param strand: A sequence of nucleotides
    """
    return dict(Counter(strand))


def validate_strand(strand: str) -> bool:
    """
    Check the strand for invalid bases
    """
    strand = strand.upper()
    count = dict(Counter(strand))
    for k in count.keys():
        if k not in NUCLEOTIDES:
            raise Exception("Invalid DNA sequence")
    return True


def most_frequent_kmer(strand: str, k: int) -> list:
    """
    Find the most common k-length sequences of nucleotides occurring in the strand.

    :param strand: A sequence of nucleotides
    :param k: Length of the k_mer
    :return: A list of k_mers
    """
    max_count = 0
    k_mers = []
    freq = defaultdict(int)
    for i in range(len(strand) - k + 1):
        freq[strand[i:i + k]] = freq[strand[i:i + k]] + 1

    for key in freq.keys():
        if freq[key] > max_count:
            max_count = freq[key]
            k_mers = [key]
        elif freq[key] == max_count:
            k_mers.append(key)
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


def transcription(strand: str) -> str:
    """
    Returns the RNA transcription of the DNA.
    """
    return strand.replace('T', 'U')


def hamming_distance(strand1: str, strand2: str) -> int:
    """
    Returns the hamming distance between the given two strands
    """
    distance = 0
    for i in range(len(strand1)):
        if strand1[i] != strand2[i]:
            distance += 1

    return distance


def skew_loc(strand: str, loc='min') -> int:
    """
    Finds the pivoting point of the G-C graph where the value stops decreasing and starts to increase.

    :param strand: A sequence of nucleotides
    :param loc: Which pivoting point is needed, ['min', 'max']
    """
    bases = {'A': 0, 'T': 0, 'C': -1, 'G': 1}
    values = [0]

    for x, y in enumerate(strand):
        values.append(values[x] + bases[y])

    values = np.array(values)
    if loc == 'min':
        return np.where(values == values.min())[0]
    else:
        return np.where(values == values.max())[0]


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
    alt_bases = {'A': 'CGT', 'C': 'AGT', 'G': 'ACT', 'T': 'ACG'}
    for dist in range(1, d + 1):
        for change_indices in combinations(range(len(k_mer)), dist):
            for substitutions in product(*[alt_bases[k_mer[i]] for i in change_indices]):
                new_mismatch = list(k_mer)
                for idx, sub in zip(change_indices, substitutions):
                    new_mismatch[idx] = sub
                mismatches.append(''.join(new_mismatch))
    return mismatches


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


def strand_score(strand, k_mer):
    length = len(k_mer)
    scores = []
    for i in range(len(strand) - length + 1):
        scores.append(hamming_distance(strand[i:i + length], k_mer))

    return min(scores)


def calculate_score(strands, k_mer):
    final_score = 0
    for strand in strands:
        final_score += strand_score(strand, k_mer)

    return final_score


def motif_enumeration(strands, k, d):
    patterns = []
    candidate = strands[0]
    for i in range(len(candidate) - k + 1):
        splice = candidate[i:i + k]
        neighbours = get_neighbors(splice, d)
        print(neighbours)
        for neighbour in neighbours:
            for strand in strands[1:]:
                if strand_score(strand, neighbour) > 1:
                    break
            else:
                patterns.append(neighbour)

    return set(patterns)


def generate_motif_profile(motifs):
    profile_loc = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    motifs = np.array([list(i) for i in motifs])
    profile = []

    for i in range(motifs.shape[1]):
        c = Counter(motifs[:, i])
        temp = [0] * 4
        for key, val in c.items():
            temp[profile_loc[key]] = val
        profile.append(temp)

    profile = np.array(profile).transpose()
    return profile / len(motifs)


def calculate_entropy(motif_profile):
    entropy_values = np.nan_to_num(np.log2(motif_profile) * motif_profile)
    return entropy_values.sum()
