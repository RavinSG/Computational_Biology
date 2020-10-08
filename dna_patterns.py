import numpy as np
import random

from params import *
from dna_replication import number_to_pattern, strand_score


def generate_motif_profile(motifs, pseudo_count=False):
    profile = {}
    motif_count = len(motifs)
    motif_len = len(motifs[0])

    for i in STR_TO_NUM.keys():
        profile[i] = [0] * motif_len

    for motif in motifs:
        for i, j in enumerate(motif):
            profile[j][i] += 1
    max_error = motif_count

    profile = np.array([x for x in profile.values()])

    if pseudo_count:
        profile += 1
        motif_count += 4
        max_error += 1

    return [profile / motif_count, sum(max_error - profile.max(axis=0))]


def consensus(motifs):
    profile = generate_motif_profile(motifs)[0]
    return "".join([NUM_TO_STR[x] for x in np.argmax(profile, axis=0)])


def calculate_entropy(motif_profile):
    entropy_values = np.nan_to_num(np.log2(motif_profile) * motif_profile)
    return entropy_values.sum()


def calculate_score(strands, k_mer):
    final_score = 0
    for strand in strands:
        final_score += strand_score(strand, k_mer)[0]

    return final_score


def median_string(strands, k):
    d = np.inf
    median_pattern = ''
    for i in range(4 ** k):
        pattern = number_to_pattern(i, k)
        score = calculate_score(strands, pattern)
        if d > score:
            median_pattern = pattern
            d = score

    return median_pattern


def most_probable_k_mer(strand: str, k: int, profile: np.ndarray):
    profile = np.array(profile)
    probabilities = calculate_k_mer_prob(strand, k, profile)
    max_loc = np.argmax(probabilities)

    return strand[max_loc: max_loc + k]


def calculate_k_mer_prob(strand, k, profile):
    probabilities = []
    indices = [STR_TO_NUM[x] for x in strand]
    for i in range(0, len(strand) - k + 1):
        indices_splice = indices[i:i + k]
        prob = 1
        for j in range(k):
            prob = prob * profile[indices_splice[j]][j]
        probabilities.append(prob)

    return np.array(probabilities)


def greedy_motif_search(strands, k, t, pseudo_count=False):
    best_motifs = [x[:k] for x in strands]
    best_score = generate_motif_profile(best_motifs)[-1]
    strand_1 = strands[0]
    for i in range(len(strands[0]) - k + 1):
        splice = strand_1[i:i + k]
        candidate_motifs = [splice]
        for strand in strands[1:]:
            profile, _ = generate_motif_profile(candidate_motifs, pseudo_count)
            best_k_mer = most_probable_k_mer(strand, k, profile)
            candidate_motifs.append(best_k_mer)
        candidate_score = generate_motif_profile(candidate_motifs)[-1]
        if best_score > candidate_score:
            best_score = candidate_score
            best_motifs = candidate_motifs

    return best_motifs


def randomized_motif_search(strands, k, t):
    random_motifs = []
    best_score = np.inf
    best_motifs = []
    len_dna = len(strands[0]) - k + 1
    for strand in strands:
        loc = random.randrange(0, len_dna)
        random_motifs.append(strand[loc:loc + k])

    profile = generate_motif_profile(random_motifs, True)[0]
    while True:
        random_motifs = [most_probable_k_mer(x, k, profile) for x in strands]
        profile, score = generate_motif_profile(random_motifs, True)
        if score < best_score:
            best_score = score
            best_motifs = random_motifs.copy()

        else:
            return best_motifs, best_score


def gibbs_sampler(strands, k, t, n):
    final_score = np.inf
    final_motifs = []
    for _ in range(50):
        motifs = []
        len_dna = len(strands[0]) - k + 1
        for strand in strands:
            loc = random.randrange(0, len_dna)
            motifs.append(strand[loc:loc + k])
        best_score = np.inf
        best_motifs = motifs.copy()
        for i in range(n):
            j = random.randint(0, t - 1)
            profile = generate_motif_profile(motifs[:j] + motifs[j + 1:], True)[0]
            probabilities = calculate_k_mer_prob(strands[j], k, profile)
            probabilities = probabilities / probabilities.sum()
            motif_loc = np.random.choice(range(len_dna), p=probabilities)
            motifs[j] = strands[j][motif_loc:motif_loc + k]
            score = generate_motif_profile(motifs)[-1]

            if score < best_score:
                best_score = score
                best_motifs = motifs.copy()

        if best_score < final_score:
            print(best_score, " ".join(best_motifs))
            final_score = best_score
            final_motifs = best_motifs.copy()

    return final_motifs
