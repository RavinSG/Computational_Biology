from params import *


def graph_of_spectrum(spectrum):
    graph = dict()

    for i, weight in enumerate(spectrum):
        graph[weight] = dict()
        for n_weight in spectrum[i + 1:]:
            gap = n_weight - weight
            if gap in REVERSE_DISTINCT:
                graph[weight][n_weight] = REVERSE_DISTINCT[gap]

    return graph
