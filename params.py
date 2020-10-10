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

EXTENDED_AMINO = {
    'G': 57, 'A': 71, 'S': 87, 'P': 97, 'V': 99, 'T': 101, 'C': 103, 'L': 113, 'N': 114,
    'D': 115, 'Q': 128, 'E': 129, 'M': 131, 'H': 137, 'F': 147, 'R': 156, 'Y': 163, 'W': 186,
    'অ': 58, 'আ': 59, 'ই': 60, 'ঈ': 61, 'উ': 62, 'ঊ': 63, 'ঋ': 64, 'ঌ': 65, 'এ': 66, 'ঐ': 67, 'ও': 68,
    'ঔ': 69, 'ক': 70, 'খ': 72, 'গ': 73, 'ঘ': 74, 'ঙ': 75, 'চ': 76, 'ছ': 77, 'জ': 78, 'ঝ': 79, 'ঞ': 80,
    'ট': 81, 'ঠ': 82, 'ড': 83, 'ঢ': 84, 'ণ': 85, 'ত': 86, 'থ': 88, 'দ': 89, 'ধ': 90, 'ন': 91, 'প': 92,
    'ফ': 93, 'ব': 94, 'ভ': 95, 'ম': 96, 'য': 98, 'র': 100, 'ল': 102, 'শ': 104, 'ষ': 105, 'স': 106,
    'হ': 107, 'ড়': 108, 'ঢ়': 109, 'য়': 110, 'ৠ': 111, 'ৡ': 112, 'ৰ': 116, 'ৱ': 117, '৳': 118, 'अ': 119,
    'आ': 120, 'इ': 121, 'ई': 122, 'उ': 123, 'ऊ': 124, 'ऋ': 125, 'ऌ': 126, 'ऍ': 127, 'ऎ': 130, 'ए': 132,
    'ऐ': 133, 'ऑ': 134, 'ऒ': 135, 'ओ': 136, 'औ': 138, 'क': 139, 'ख': 140, 'ग': 141, 'घ': 142, 'ङ': 143,
    'च': 144, 'छ': 145, 'ज': 146, 'झ': 148, 'ञ': 149, 'ट': 150, 'ठ': 151, 'ड': 152, 'ढ': 153, 'ण': 154,
    'त': 155, 'थ': 157, 'द': 158, 'ध': 159, 'न': 160, 'ऩ': 161, 'प': 162, 'फ': 164, 'ब': 165, 'भ': 166,
    'म': 167, 'य': 168, 'र': 169, 'ऱ': 170, 'ल': 171, 'ळ': 172, 'ऴ': 173, 'व': 174, 'श': 175, 'ष': 176,
    'स': 177, 'ह': 178, 'क़': 179, 'ख़': 180, 'ग़': 181, 'ज़': 182, 'ड़': 183, 'ढ़': 184, 'फ़': 185, 'य़': 187,
    'ॠ': 188, 'ॡ': 189, 'અ': 190, 'આ': 191, 'ઇ': 192, 'ઈ': 193, 'ઉ': 194, 'ઊ': 195, 'ઋ': 196, 'એ': 197,
    'ઐ': 198, 'ઓ': 199, 'ઔ': 200
}
