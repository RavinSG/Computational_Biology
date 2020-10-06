from collections import defaultdict, Counter

NUCLEOTIDES = ['A', 'T', 'C', 'G']
COMPLIMENTS = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}


def count_nucleotides(strand: str) -> dict:
    """
    Returns a dictionary containing the number of times each nucleotide is present in the strand.

    :param strand: A sequence of nucleotides
    """
    return dict(Counter(strand))


def validate_strand(strand: str) -> bool:
    """
    Check the strand for invalid bases
    """
    strand = strand.upper()
    count = dict(Counter(strand))
    for k in count.keys():
        if k not in NUCLEOTIDES:
            raise Exception("Invalid DNA sequence")
    return True


def transcription(strand: str) -> str:
    """
    Returns the RNA transcription of the DNA.
    """
    return strand.replace('T', 'U')


def coloured(seq):
    base_colours = {
        "A": '\033[92m',
        'C': '\033[94m',
        'G': '\033[93m',
        'T': '\033[91m',
        'U': '\033[91m',
        'reset': '\033[0;0m'
    }

    temp_str = ""

    for nuc in seq:
        if nuc in base_colours:
            temp_str += base_colours[nuc] + nuc
        else:
            temp_str += base_colours['reset'] + nuc

    return temp_str + '\033[0;0m'
