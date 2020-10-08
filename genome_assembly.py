from collections import defaultdict


def composition(strand, k):
    reads = [strand[x:x + k] for x in range(len(strand) - k + 1)]
    reads.sort()
    return reads


def simple_string_assembly(strands):
    start = strands[0]
    start = start + ''.join(x[-1] for x in strands[1:])
    return start


def simple_overlapping_graph(strands):
    prefixes = defaultdict(list)
    for i in strands:
        prefixes[i[:-1]].append(i)

    for i in strands:
        strings = prefixes[i[1:]]
        if len(strings) > 0:
            print(i, '->', ",".join(strings))
