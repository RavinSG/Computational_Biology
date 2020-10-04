from collections import defaultdict


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


def get_complement(strand, reverse=False):
    pairs = {
        'A': 'T',
        'T': 'A',
        'G': 'C',
        'C': 'G'
    }
    complement = ''.join(pairs[x] for x in strand)
    if reverse:
        complement = complement[::-1]

    return complement


def pattern_matching(strand, pattern):
    positions = []
    start = 0
    while True:
        start = strand.find(pattern, start)
        print(start)
        if start != -1:
            positions.append(start)
            start += 1
        else:
            return positions
