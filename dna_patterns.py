import numpy as np
from collections import Counter


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
