import numpy as np
from collections import Counter

from params import *
from dna_replication import hamming_distance, number_to_pattern


def generate_motif_profile(motifs):
    motifs = np.array([list(i) for i in motifs])
    profile = []

    for i in range(motifs.shape[1]):
        c = Counter(motifs[:, i])
        temp = [0] * 4
        for key, val in c.items():
            temp[STR_TO_NUM[key]] = val
        profile.append(temp)

    profile = np.array(profile).transpose()
    return [profile / len(motifs), sum(motifs.shape[0] - profile.max(axis=0))]


def consensus(motifs):
    profile = generate_motif_profile(motifs)[0]
    return "".join([NUM_TO_STR[x] for x in np.argmax(profile, axis=0)])


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
