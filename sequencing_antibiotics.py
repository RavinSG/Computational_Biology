from math import log, exp
from collections import Counter

from params import *
from utilities import transcription
from dna_replication import get_complement


def protein_translation(rna_strand):
    protein = ''
    for i in range(0, len(rna_strand), 3):
        try:
            protein += CODON_TABLE[rna_strand[i:i + 3]]
        except TypeError:
            return protein

    return protein


def peptide_encoding_substring(strand, peptide):
    str_len = 3 * len(peptide)
    substrings = []
    for i in range(len(strand) - str_len + 1):
        substring = strand[i:i + str_len]
        complement = get_complement(substring, reverse=True)

        for s in [substring, complement]:
            rna = transcription(s)
            if protein_translation(rna) == peptide:
                substrings.append(substring)
                break
    return substrings


def theoretical_spectrum(peptide, linear=False):
    cyclospectrum = ["", peptide]
    weights = [0, sum([INTEGER_MASS[x] for x in peptide])]

    l_peptide = len(peptide)

    if linear:
        for i in range(1, l_peptide):
            for j in range(l_peptide - i + 1):
                codons = peptide[j:j + i]
                weights.append(sum([INTEGER_MASS[x] for x in codons]))
                cyclospectrum.append(codons)
    else:
        peptide = peptide * 2
        for i in range(1, l_peptide):
            for j in range(l_peptide):
                codons = peptide[j:j + i]
                weights.append(sum([INTEGER_MASS[x] for x in codons]))
                cyclospectrum.append(codons)

    weights, cyclospectrum = zip(*sorted(zip(weights, cyclospectrum), key=lambda pair: pair[0]))

    return weights, cyclospectrum


def mass_peptide(peptide):
    return sum([INTEGER_MASS[x] for x in peptide])


def num_of_possible_peptides(mass, mass_dict):
    if mass == 0:
        return 1, mass_dict
    elif mass < 57:
        return 0, mass_dict
    elif mass in mass_dict:
        return mass_dict[mass], mass_dict
    else:
        n = 0
        for m in DISTINCT_MASSES.values():
            i, mass_dict = num_of_possible_peptides(mass - m, mass_dict)
            n += i
        mass_dict[mass] = n

        return n, mass_dict


def calculate_c(num_1, num_2):
    i_1 = num_of_possible_peptides(num_1, {})[0]
    i_2 = num_of_possible_peptides(num_2, {})[0]

    c = exp(log(i_1 / i_2) / (num_1 - num_2))

    print(c)


def expand_peptide(peptide):
    return [peptide + x for x in DISTINCT_MASSES.keys()]


def check_consistency(spectrum_1, spectrum_2, mirror=False):
    # t_w_spectrum = theoretical_spectrum(peptide)[0]
    # spectrum_2 = Counter(t_w_spectrum)
    if mirror:
        if spectrum_1 != spectrum_2:
            return False
    else:
        for weight, count in spectrum_2.items():
            if weight in spectrum_1:
                if count <= spectrum_1[weight]:
                    continue
            return False

    return True


def cyclopeptide_sequencing(spectrum):
    weight_counts = Counter(spectrum)
    parent_weight = spectrum[-1]
    final_peptides = set()
    candidate_peptides = [""]

    while len(candidate_peptides) > 0:
        previous_batch = candidate_peptides.copy()
        candidate_peptides = []

        while previous_batch:
            candidate_peptides += expand_peptide(previous_batch.pop())
        next_batch = candidate_peptides.copy()

        for peptide in candidate_peptides:
            if mass_peptide(peptide) == parent_weight:
                t_w_spectrum = theoretical_spectrum(peptide)[0]
                t_w_count = Counter(t_w_spectrum)

                if check_consistency(weight_counts, t_w_count, mirror=True):
                    next_batch.remove(peptide)
                    final_peptides.add(peptide)
            else:
                t_w_spectrum = theoretical_spectrum(peptide, linear=True)[0]
                t_w_count = Counter(t_w_spectrum)

                if check_consistency(weight_counts, t_w_count):
                    continue
                else:
                    next_batch.remove(peptide)

        candidate_peptides = next_batch
    return final_peptides
