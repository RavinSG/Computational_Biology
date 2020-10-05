import numpy as np
from collections import defaultdict, Counter
from itertools import combinations, product

NUCLEOTIDES = ['A', 'T', 'C', 'G']
COMPLIMENTS = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}


def count_nucleotides(strand):
    return dict(Counter(strand))


def validate_strand(strand):
    strand = strand.upper()
    count = dict(Counter(strand))
    for k in count.keys():
        if k not in NUCLEOTIDES:
            raise Exception("Invalid DNA sequence")
    return True


def most_frequent_kmer(strand, k):
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


def get_complement(strand, reverse=True):
    complement = ''.join(COMPLIMENTS[x] for x in strand)
    if reverse:
        complement = complement[::-1]

    return complement


def pattern_matching(strand, pattern):
    positions = []
    start = 0
    while True:
        start = strand.find(pattern, start)
        if start != -1:
            positions.append(start)
            start += 1
        else:
            return positions


def transcription(strand):
    return strand.replace('T', 'U')


def hamming_distance(strand1, strand2):
    distance = 0
    for i in range(len(strand1)):
        if strand1[i] != strand2[i]:
            distance += 1

    return distance


def skew_loc(strand, loc='min'):
    bases = {'A': 0, 'T': 0, 'C': -1, 'G': 1}
    values = [0]

    for x, y in enumerate(strand):
        values.append(values[x] + bases[y])

    values = np.array(values)
    if loc == 'min':
        return np.where(values == values.min())[0]
    else:
        return np.where(values == values.max())[0]


def approximate_pattern_count(strand, pattern, d):
    s_l = len(strand)
    p_l = len(pattern)
    count = 0
    for i in range(s_l - p_l + 1):
        splice = strand[i: i + p_l]
        if (hamming_distance(splice, pattern)) <= d:
            count += 1

    return count


def get_neighbors(k_mer, d):
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


def frequent_words_with_mismatch(strand, k, d):
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
