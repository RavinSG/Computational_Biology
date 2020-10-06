import numpy as np
from collections import Counter
from dna_replication import hamming_distance, number_to_pattern


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
    return [profile / len(motifs), sum(motifs.shape[0] - profile.max(axis=0))]


def consensus(motifs):
    mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    profile = generate_motif_profile(motifs)[0]
    return "".join([mapping[x] for x in np.argmax(profile, axis=0)])


def calculate_entropy(motif_profile):
    entropy_values = np.nan_to_num(np.log2(motif_profile) * motif_profile)
    return entropy_values.sum()


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


def median_string(strands, k):
    d = np.inf
    patterns = []
    for i in range(4 ** k):
        pattern = number_to_pattern(i, k)
        score = calculate_score(strands, pattern)
        if d > score:
            patterns.append(pattern)
            d = score

    return patterns
