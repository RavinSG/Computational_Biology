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

    for i in range(len(strand) - str_len + 1):
        substring = strand[i:i + str_len]
        complement = get_complement(substring, reverse=True)

        for s in [substring, complement]:
            rna = transcription(s)
            print(protein_translation(rna))


peptide_encoding_substring("ATGGCCAT", "MA")
