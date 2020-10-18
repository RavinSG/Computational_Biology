from math import log, exp
from collections import Counter, defaultdict

from params import *
from utilities import transcription
from dna_replication import get_complement

DISTINCT_DICT = DISTINCT_MASSES
MASS_DICT = INTEGER_MASS


def protein_translation(rna_strand: str) -> str:
    """
    Given a RNA strand, returns the protein that is translated by the strand.

    :param rna_strand: A sequence of RNA bases
    :return: An amino acid sequence
    """
    protein = ''
    for i in range(0, len(rna_strand), 3):
        try:
            protein += CODON_TABLE[rna_strand[i:i + 3]]
        except TypeError:
            return protein

    return protein


def peptide_encoding_substring(strand: str, peptide: str) -> list:
    """
    Given a peptide and a strand, finds the substrings that transcript in to RNA an then translates into the peptide.
    Since a DNA strand is a double-helix, the reverse compliment of each substring is also checked.

    :param strand: A sequence of nucleotides
    :param peptide: A sequence of amino acids
    :return: A list of substrings that translate into the peptide
    """
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


def theoretical_spectrum(peptide: str, linear=False) -> tuple:
    """
    Given a peptide returns the theoretical spectrum of the peptide along with the weights of the sub-peptides.
    A theoretical spectrum represents all the possible sub-peptides for a given peptide including the empty peptide and
    the given peptide itself.

    A sub-peptide is a linear fragment of a larger peptide. If the original peptide is circular, the sub-peptides will
    contain peptides that wrap around the original peptide.

    :param peptide: A sequence of amino acids
    :param linear: If true some sub-peptides will wrap around the original peptide
    :return: A list of all possible sub-peptides and their weights
    """
    cyclospectrum = ["", peptide]
    weights = [0, sum([MASS_DICT[x] for x in peptide])]

    l_peptide = len(peptide)

    if linear:
        for i in range(1, l_peptide):
            for j in range(l_peptide - i + 1):
                codons = peptide[j:j + i]
                weights.append(sum([MASS_DICT[x] for x in codons]))
                cyclospectrum.append(codons)
    else:
        peptide = peptide * 2
        for i in range(1, l_peptide):
            for j in range(l_peptide):
                codons = peptide[j:j + i]
                weights.append(sum([MASS_DICT[x] for x in codons]))
                cyclospectrum.append(codons)

    weights, cyclospectrum = zip(*sorted(zip(weights, cyclospectrum), key=lambda pair: pair[0]))

    return weights, cyclospectrum


def mass_peptide(peptide: list) -> list:
    """
    Given a list of peptides calculate the masses.
    """
    return sum([MASS_DICT[x] for x in peptide])


def num_of_possible_peptides(mass: int, mass_dict: dict) -> tuple:
    """
    Given a mass and a mass spectrum, calculate the number of all possible linear peptides with the given mass that
    coincides with the mass spectrum.

    :param mass: Possible mass of a peptide
    :param mass_dict: A weight spectrum
    :return: The number of possible peptides and a dictionary containing the number of peptides for each mass entry
    """
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


def calculate_c(num_1: int, num_2: int) -> float:
    """
    Calculate the slope of the Integer Mass Vs number of peptides graph
    """
    i_1, _ = num_of_possible_peptides(num_1, {})
    i_2, _ = num_of_possible_peptides(num_2, {})

    c = exp(log(i_1 / i_2) / (num_1 - num_2))

    return c


def expand_peptide(peptide: str) -> list:
    """
    Returns all peptides that can be created by appending one amino acid to the peptide.
    """
    return [peptide + x for x in DISTINCT_DICT.keys()]


def check_consistency(spectrum_1: dict, spectrum_2: dict, mirror=False) -> bool:
    """
    Given two spectra, checks whether the second spectrum is consistent with the first spectrum.

    Spectrum A is consistent with Spectrum B if, for all keys in Spectrum A, the value of Spectrum A is lower than or
    equal to the value of Spectrum B.

    :param spectrum_1: Parent mass spectrum
    :param spectrum_2: Child mass spectrum
    :param mirror: If true both spectra should be identical
    :return: Consistency of the two spectra
    """
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


def cyclopeptide_sequencing(spectrum: list) -> set:
    """
    Given a theoretical mass spectrum, finds all the peptides that produce the mass spectrum.
    :param spectrum: A theoretical mass spectrum
    :return: The peptides responsible for the spectrum
    """
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
            # If the mass of the peptide is equal to the highest mass in the spectrum, check for spectrum equivalence
            # since it can't be expanded further while being consistent with the spectrum
            if mass_peptide(peptide) == parent_weight:
                t_w_spectrum, _ = theoretical_spectrum(peptide)
                t_w_count = Counter(t_w_spectrum)

                if check_consistency(weight_counts, t_w_count, mirror=True):
                    next_batch.remove(peptide)
                    final_peptides.add(peptide)
            else:
                t_w_spectrum, _ = theoretical_spectrum(peptide, linear=True)
                t_w_count = Counter(t_w_spectrum)

                if check_consistency(weight_counts, t_w_count):
                    continue
                else:
                    next_batch.remove(peptide)

        candidate_peptides = next_batch
    return final_peptides


def score_cyclopeptide(peptide: str, experimental_spectrum: list, linear=True) -> int:
    """
    Calculates a score of a peptide against an experimental spectrum which was obtained by an experiment.

    Initially the theoretical weight spectrum of the peptide is calculated. The score is calculated by counting the
    number of masses shared by the two spectra.

    :param peptide: The candidate peptide
    :param experimental_spectrum: A weight spectrum
    :param linear: If ture only linear peptides are used to calculate the theoretical weight spectrum
    :return: An integer score
    """
    t_spectrum, _ = theoretical_spectrum(peptide, linear)
    t_spectrum = Counter(t_spectrum)
    e_spectrum = defaultdict(int, Counter(experimental_spectrum))
    score = 0
    for i, j in t_spectrum.items():
        score += min(j, e_spectrum[i])

    return score


def trim_leaderboard(leaderboard: list, spectrum: list, n: int) -> list:
    """
    Given a list of peptides as a leaderboard and a spectrum, returns the top n peptides scored against the spectrum
    with all ties.

    :param leaderboard: A list of peptides
    :param spectrum: A mass spectrum
    :param n: Leaderboard cutoff value
    :return: Top n entries
    """
    scores = [score_cyclopeptide(x, spectrum) for x in leaderboard]
    scores, leaderboard = zip(*sorted(zip(scores, leaderboard), key=lambda pair: pair[0], reverse=True))

    if not len(leaderboard) <= n:
        try:
            while scores[n] == scores[n + 1]:
                n += 1
        except IndexError:
            n = n - 1

    return list(leaderboard[:n])


def leaderboard_cyclopeptide_sequencing(spectrum: list, n: int, weight_dict: dict = None) -> tuple:
    """
    Given a mass spectrum and an integer n, use a scoreboard to find the most probable peptide that would produce a mass
    spectrum to the given spectrum.

    :param spectrum: A mass spectrum
    :param n: Threshold number for the scoreboard
    :param weight_dict: A weight dictionary to be used for mass calculation
    :return: A tuple containing a peptide of the highest score and a list containing all peptides with the highest score
    """
    global DISTINCT_DICT, MASS_DICT
    if weight_dict is not None:
        DISTINCT_DICT = weight_dict
        MASS_DICT = weight_dict

    parent_mass = spectrum[-1]
    leaderboard = [""]
    leader_peptide = ''
    leader_score = score_cyclopeptide(leader_peptide, spectrum)
    best_peps = []

    while True:
        previous_batch = leaderboard.copy()
        leaderboard = []
        while previous_batch:
            leaderboard += expand_peptide(previous_batch.pop())
        next_batch = leaderboard.copy()

        for peptide in leaderboard:
            peptide_mass = mass_peptide(peptide)
            if peptide_mass == parent_mass:
                peptide_score = score_cyclopeptide(peptide, spectrum, linear=False)

                if peptide_score > leader_score:
                    leader_peptide = peptide
                    leader_score = peptide_score
                    next_batch.remove(peptide)
                    best_peps = [peptide]
                    # Append the peptide to the peptide list with highest scores
                elif peptide_score == leader_score:
                    best_peps.append(peptide)
            elif peptide_mass > parent_mass:
                next_batch.remove(peptide)

        if len(next_batch) > 0:
            leaderboard = trim_leaderboard(next_batch, spectrum, n)

        else:
            # Revert the weight dictionaries back to their default values
            DISTINCT_DICT = DISTINCT_MASSES
            MASS_DICT = INTEGER_MASS
            return leader_peptide, best_peps


def spectral_convolution(spectrum: list) -> list:
    """
    Given a spectrum calculate weight differences between all pairs. If there are multiple values of the same mass, each
     mass will be treated as an individual value.
    :param spectrum: A mass spectrum
    :return: A list of masses except 0
    """
    convolution = []
    for i in range(len(spectrum) - 1):
        for j in range(i, len(spectrum)):
            convolution.append(spectrum[j] - spectrum[i])

    return [x for x in convolution if x != 0]


def convolution_cyclopeptide_sequencing(spectrum: list, m: int, n: int) -> tuple:
    """
    Given a mass spectrum, identify candidate amino acids based on the convolution spectrum and use only these candidate
    amino acids as the starting peptides. The top m candidates will be selected and a scoreboard with a threshold of m
    is used.

    :param spectrum: A mass spectrum
    :param m: Number of peptide candidates
    :param n: Scoreboard threshold
    :return: A tuple containing a peptide of the highest score and a list containing all peptides with the highest score
    """
    spectrum.sort()
    convolution = Counter(spectral_convolution(spectrum))
    scores, amino_acids = list(zip(*sorted(zip(convolution.values(), convolution.keys()), reverse=True)))

    alphabet = {}
    i = 0
    while m > 0:
        try:
            codon = REVERSE_EXTENDED[amino_acids[i]]
            alphabet[codon] = EXTENDED_AMINO[codon]
            m -= 1
        except KeyError:
            pass
        finally:
            i += 1

    try:
        while scores[i] == scores[i + 1]:
            codon = REVERSE_EXTENDED[amino_acids[i]]
            alphabet[codon] = EXTENDED_AMINO[codon]
            i += 1
    except (KeyError, IndexError):
        pass

    print(alphabet)
    pep, b_peps = leaderboard_cyclopeptide_sequencing(spectrum, n, alphabet)

    return pep, b_peps
