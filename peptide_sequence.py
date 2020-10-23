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


def peptide_to_vector(peptide):
    vector = []
    for i in peptide:
        pep_vector = [0] * INTEGER_MASS[i]
        pep_vector[-1] = 1
        vector += pep_vector

    return vector


def vector_to_peptide(vector):
    vector = np.array(vector)
    positions = np.where(vector == 1)[0] + 1
    shift_positions = np.concatenate(([0], positions[:-1]))
    positions = positions - shift_positions
    peptide = ""
    for pos in positions:
        peptide += REVERSE_DISTINCT[pos]

    print(peptide)


def max_score_peptide(vector_spectrum):
    vector_spectrum = [0] + vector_spectrum
    num_nodes = len(vector_spectrum)
    nodes = dict()
    for i in range(num_nodes):
        children = dict()
        for weight in REVERSE_DISTINCT:
            dst_node = weight + i
            if dst_node >= num_nodes:
                break
            # print(dst_node)
            children[dst_node] = vector_spectrum[dst_node]

        if len(children) > 0:
            nodes[i] = children

    node_values = {x: [-np.inf, -1] for x in range(num_nodes)}
    node_values[0] = [0, -1]

    for parent, children in nodes.items():
        node_val = node_values[parent][0]
        for child, edge in children.items():
            path_value = node_val + edge
            if node_values[child][0] < path_value:
                node_values[child] = [path_value, parent]

    peptide = ''
    node = num_nodes - 1
    while node > 0:
        weight = node - node_values[node][-1]
        peptide += REVERSE_DISTINCT[weight]
        node = node_values[node][-1]

    return peptide[::-1]
