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
