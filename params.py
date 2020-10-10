NUCLEOTIDES = ['A', 'T', 'C', 'G']
COMPLIMENTS = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
SWAP_BASES = {'A': 'CGT', 'C': 'AGT', 'G': 'ACT', 'T': 'ACG'}

GC_SCORE = {'A': 0, 'T': 0, 'C': -1, 'G': 1}

STR_TO_NUM = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
NUM_TO_STR = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

CODON_TABLE = {
    'AAA': 'K', 'AAC': 'N', 'AAG': 'K', 'AAU': 'N', 'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACU': 'T', 'AGA': 'R',
    'AGC': 'S', 'AGG': 'R', 'AGU': 'S', 'AUA': 'I', 'AUC': 'I', 'AUG': 'M', 'AUU': 'I', 'CAA': 'Q', 'CAC': 'H',
    'CAG': 'Q', 'CAU': 'H', 'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCU': 'P', 'CGA': 'R', 'CGC': 'R', 'CGG': 'R',
    'CGU': 'R', 'CUA': 'L', 'CUC': 'L', 'CUG': 'L', 'CUU': 'L', 'GAA': 'E', 'GAC': 'D', 'GAG': 'E', 'GAU': 'D',
    'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCU': 'A', 'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGU': 'G', 'GUA': 'V',
    'GUC': 'V', 'GUG': 'V', 'GUU': 'V', 'UAA': 0, 'UAC': 'Y', 'UAG': 0, 'UAU': 'Y', 'UCA': 'S', 'UCC': 'S',
    'UCG': 'S', 'UCU': 'S', 'UGA': 0, 'UGC': 'C', 'UGG': 'W', 'UGU': 'C', 'UUA': 'L', 'UUC': 'F', 'UUG': 'L',
    'UUU': 'F',
}

INTEGER_MASS = {
    'G': 57, 'A': 71, 'S': 87, 'P': 97, 'V': 99, 'T': 101, 'C': 103, 'I': 113, 'L': 113, 'N': 114,
    'D': 115, 'K': 128, 'Q': 128, 'E': 129, 'M': 131, 'H': 137, 'F': 147, 'R': 156, 'Y': 163, 'W': 186,
}

DISTINCT_MASSES = {'G': 57, 'A': 71, 'S': 87, 'P': 97, 'V': 99, 'T': 101, 'C': 103, 'L': 113, 'N': 114, 'D': 115,
                   'Q': 128, 'E': 129, 'M': 131, 'H': 137, 'F': 147, 'R': 156, 'Y': 163, 'W': 186}
