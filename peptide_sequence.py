from params import *

from collections import defaultdict


def graph_of_spectrum(spectrum):
    graph = dict()

    for i, weight in enumerate(spectrum):
        graph[weight] = dict()
        for n_weight in spectrum[i + 1:]:
            gap = n_weight - weight
            if gap in REVERSE_DISTINCT:
                graph[weight][n_weight] = REVERSE_DISTINCT[gap]

    return graph


def ideal_spectrum(peptide):
    spectrum = []

    for i in range(len(peptide)):
        prefix = peptide[:i]
        suffix = peptide[i:]

        pre_weight = sum([INTEGER_MASS[x] for x in prefix])
        suf_weight = sum([INTEGER_MASS[x] for x in suffix])

        spectrum += [pre_weight, suf_weight]

    spectrum.sort()
    return spectrum[1:]


def graph_reversal(graph):
    reversed_graph = defaultdict(dict)
    for key in graph:
        for child in graph[key]:
            reversed_graph[child][key] = graph[key][child]

    return reversed_graph


def decode_ideal_spectrum(spectrum):
    graph = graph_of_spectrum(spectrum)
    reverse_graph = graph_reversal(graph)

    end_node = spectrum[-1]
    peptide = ""

    peptide = construct_peptide(reverse_graph, end_node, peptide, spectrum[1:])
    return peptide


def construct_peptide(graph, node, peptide, spectrum):
    if node == 0:
        candidate_spectrum = ideal_spectrum(peptide)
        if candidate_spectrum == spectrum:
            # print(peptide)
            return peptide[::-1]

    else:
        for parent in graph[node]:
            parent_peptide = peptide + graph[node][parent]
            found_pep = construct_peptide(graph, parent, parent_peptide, spectrum)
            if found_pep is not None:
                return found_pep
