import random
import numpy as np

from params import *
from dna_replication import number_to_pattern, strand_score


def generate_motif_profile(motifs: list, pseudo_count=False) -> list:
    """
    Given a list of motifs of length k, generates a 4xk matrix where each row denotes a nucleotide and the i-th
    column represents the percentage of each nucleotide in the i-th position of all motifs.

    :param motifs: A list of equal length motifs
    :param pseudo_count: If true Cromwell's rule will be applies when calculating probabilities/percentages
    :return: A list containing the matrix and score for the list of motifs relative to the consensus string.
    """
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


def consensus(motifs: list) -> str:
    profile, _ = generate_motif_profile(motifs)
    return "".join([NUM_TO_STR[x] for x in np.argmax(profile, axis=0)])


def calculate_entropy(motif_profile: list) -> float:
    """
    Given a motif profile with probabilities, the cumulative information entropy is calculated.

    :param motif_profile: A 4 x k matrix
    :return: Sum of information entropy of across the columns
    """
    entropy_values = np.nan_to_num(np.log2(motif_profile) * motif_profile)
    return entropy_values.sum()


def calculate_score(strands: list, k_mer: str) -> int:
    """
    Given a list of dna strands and a reference k_mer, the score of the set of strands with respect to the k_mer is
    calculated.

    :param strands: A list of dna strands
    :param k_mer: The reference k_mer the score should be calculated against
    :return: An integer reflecting the score
    """
    final_score = 0
    for strand in strands:
        final_score += strand_score(strand, k_mer)[0]

    return final_score


def median_string(strands: list, k: int) -> str:
    """
    Given a set of dna strands and an integer k, find the k_mer that has the lowest score with respect to the list of
    strands.

    :param strands: A list of dna strands
    :param k: The length of the median string
    :return: The median string of length k
    """
    d = np.inf
    median_pattern = ''

    for i in range(4 ** k):
        pattern = number_to_pattern(i, k)
        score = calculate_score(strands, pattern)
        if d > score:
            median_pattern = pattern
            d = score

    return median_pattern


def calculate_k_mer_prob(strand: str, k: int, profile: np.ndarray) -> np.ndarray:
    """
    Given a dna strand and a motif profile, this function will return an array of probabilities where the i-th element
    represents the probability of strand[i:i+k] occurring with respect to the probabilities given by the profile.

    :param strand: A list of dna strands
    :param k: Length of the k_mer
    :param profile: A motif profile with probabilities
    :return: An array of length len(strand)+1-k with the probabilities.
    """
    probabilities = []
    indices = [STR_TO_NUM[x] for x in strand]

    for i in range(0, len(strand) - k + 1):
        indices_splice = indices[i:i + k]
        prob = 1
        for j in range(k):
            prob = prob * profile[indices_splice[j]][j]
        probabilities.append(prob)

    return np.array(probabilities)


def most_probable_k_mer(strand: str, k: int, profile: np.ndarray) -> str:
    """
    Finds the most probable k_mer in the strand with respect to the given profile..

    :param strand: A sequence of nucleotides
    :param k: Length of the k_mer
    :param profile: A motif profile with probabilities
    :return: The most probable sequence of nucleotides to occur with respect to the profile
    """
    profile = np.array(profile)
    probabilities = calculate_k_mer_prob(strand, k, profile)
    max_loc = np.argmax(probabilities)

    return strand[max_loc: max_loc + k]


def greedy_motif_search(strands, k, t, pseudo_count=False):
    """
    Given a list of dna strands, finds a k_mer in a greedy approach which has the highest chance of appearing in the
    list strands.

    :param strands: A list of dna strands
    :param k: Length of the k_mer
    :param t: Number of strands
    :param pseudo_count: If true Cromwell's rule will be applies when calculating probabilities/percentages
    :return: A list of k_mers of length t
    """

    # Initialize the fist set of k_mers in the strings as the best_motifs found and calculate the score
    best_motifs = [x[:k] for x in strands]
    _, best_score = generate_motif_profile(best_motifs)
    strand_1 = strands[0]

    # Move a sliding window of length k along the first strand to select the base k_mer
    for i in range(len(strands[0]) - k + 1):
        splice = strand_1[i:i + k]
        candidate_motifs = [splice]
        # For every other strand find the most probable k_mer using the motifs generated till the previous strand
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
    """
    Given a set of dna strands, uses a Monte-Carlo simulation to find the best set of k_mers.

    :param strands: A list of dna strands
    :param k: Length of the k_mer
    :param t: Number of strands
    :return: A list of k_mers of length t
    """
    random_motifs = []
    best_score = np.inf
    best_motifs = []
    len_dna = len(strands[0]) - k + 1
    # Randomly pick a set of k_mers, one from each dna strand
    for strand in strands:
        loc = random.randrange(0, len_dna)
        random_motifs.append(strand[loc:loc + k])

    # Until the score stops increasing, use the randomly selected strands to generate a new motif profile, use the
    # generated profile to find a new set of k_mers and calculate the new score.
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
    """
    Same as randomized_motif_search, but instead of selecting a whole new set of k_mers for the next iteration,
    randomly selects one strand and search for a new k_mer in that strand respective to the profile.

    :param strands: A list of dna strands
    :param k: Length of the k_mer
    :param t: Number of strands
    :param n: Number of times the iteration process should be done
    :return: A list of k_mers of length t
    """
    final_score = np.inf
    final_motifs = []
    # Number of times the whole process should be carried out
    for _ in range(50):
        motifs = []
        len_dna = len(strands[0]) - k + 1
        for strand in strands:
            loc = random.randrange(0, len_dna)
            motifs.append(strand[loc:loc + k])
        best_score = np.inf
        best_motifs = motifs.copy()
        for i in range(n):
            # Select a random strand from [Strand_1, Strand_2,...., Strand_t]
            j = random.randint(0, t - 1)
            profile = generate_motif_profile(motifs[:j] + motifs[j + 1:], True)[0]
            probabilities = calculate_k_mer_prob(strands[j], k, profile)
            probabilities = probabilities / probabilities.sum()
            # Instead of selecting the most probable k_mer, probabilistically selects the best k_mer
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
