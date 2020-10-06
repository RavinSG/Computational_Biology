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

    return [min(scores), np.argmin(scores)]


def calculate_score(strands, k_mer):
    final_score = 0
    for strand in strands:
        final_score += strand_score(strand, k_mer)[0]

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


def most_probable_k_mer(strand, k, profile):
    profile = np.array(profile)
    probable_k_mer = strand[:k]
    max_prob = 0
    for i in range(0, len(strand) - k + 1):
        splice = strand[i:i + k]
        indices = [STR_TO_NUM[x] for x in splice]
        prob = np.prod(profile[[indices], range(len(indices))])
        if prob > max_prob:
            max_prob = prob
            probable_k_mer = splice

    return probable_k_mer


def greedy_motif_search(strands, k, t):
    best_motifs = [x[:k] for x in strands]
    best_score = generate_motif_profile(best_motifs)[-1]
    strand_1 = strands[0]
    for i in range(len(strands[0]) - k + 1):
        splice = strand_1[i:i + k]
        candidate_motifs = [splice]
        for strand in strands[1:]:
            profile, _ = generate_motif_profile(candidate_motifs)
            best_k_mer = most_probable_k_mer(strand, k, profile)
            candidate_motifs.append(best_k_mer)
        candidate_score = generate_motif_profile(candidate_motifs)[-1]
        if best_score > candidate_score:
            best_score = candidate_score
            best_motifs = candidate_motifs

    return best_motifs
