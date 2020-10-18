import numpy as np
from collections import defaultdict

"""
(+1, -2, -3, +4)        - genome
(2, 3), (4, 5), (6, 1)  - coloured edges/ graph
(1, 2, 3, 4, 5, 6)      - node sequence/ cycle
"""


def greedy_sort(perm):
    k = len(perm)
    perm = np.array(perm)
    steps = 0
    for i in range(k):
        if perm[i] != i + 1:
            steps += 1
            k_index = np.where(np.abs(perm) == i + 1)[0][0]
            perm = np.concatenate((perm[:i], (-1 * perm[i:k_index + 1][::-1]), perm[k_index + 1:]))
            print(" ".join(f'{x:+d}' for x in perm))
        if perm[i] != i + 1:
            steps += 1
            perm[i] = abs(i + 1)
            print(" ".join(f'{x:+d}' for x in perm))

    return perm, steps


def count_breakpoints(perm):
    k = len(perm)
    points = 0
    for i in range(k - 1):
        if perm[i + 1] - perm[i] != 1:
            points += 1
    if perm[0] != 1:
        points += 1
    if perm[-1] != k:
        points += 1

    return points


def genome_to_cycle(chromosome):
    cycle = []
    k = len(chromosome)
    for i in range(1, k + 1):
        j = chromosome[i - 1]
        j = j * 2
        if j > 0:
            cycle += [j - 1, j]
        else:
            cycle += [-j, -j - 1]
    return cycle


def cycle_to_genome(cycle):
    chromosome = []
    k = len(cycle)
    for i in range(0, k, 2):
        head, tail = cycle[i:i + 2]
        if head < tail:
            chromosome.append(tail // 2)
        else:
            chromosome.append(-head // 2)
    return chromosome


def coloured_edges(chromosomes):
    edges = []
    for chromosome in chromosomes:
        cycle = genome_to_cycle(chromosome)
        k = len(cycle)
        for i in range(1, k - 2, 2):
            edges.append((cycle[i], cycle[i + 1]))

        edges.append((cycle[-1], cycle[0]))

    return edges


def graph_to_cycles(graph, breakpoint_graph=False):
    nodes = defaultdict(list)

    for i in graph:
        nodes[i[0]].append(i[1])
        nodes[i[1]].append(i[0])

    if not breakpoint_graph:
        for i in nodes.keys():
            nodes[i].append(i + 1 if i % 2 == 1 else i - 1)

    chromosomes = []
    prev_node = list(nodes.keys())[0]
    next_node = nodes[prev_node][0]
    nodes[prev_node].remove(next_node)
    nodes[next_node].remove(prev_node)
    chromosome = [next_node]
    prev_node = next_node
    while True:
        try:
            next_node = nodes[prev_node][0]
            chromosome.append(next_node)
            nodes[prev_node].remove(next_node)
            nodes[next_node].remove(prev_node)
            nodes.pop(prev_node)
            prev_node = next_node
        except IndexError:
            chromosomes.append(chromosome)
            nodes.pop(prev_node)
            if len(nodes) == 0:
                break
            else:
                prev_node = list(nodes.keys())[0]
                next_node = nodes[prev_node][0]
                nodes[prev_node].remove(next_node)
                nodes[next_node].remove(prev_node)
                chromosome = [next_node]
                prev_node = next_node
    return chromosomes


def two_break_on_graph(graph, a, b, c, d):
    if (a, b) in graph:
        graph.pop(graph.index((a, b)))
    else:
        graph.pop(graph.index((b, a)))
    if (c, d) in graph:
        graph.pop(graph.index((c, d)))
    else:
        graph.pop(graph.index((d, c)))

    graph += [(a, c), (b, d)]

    return graph


def two_break_on_genome(genome, a, b, c, d):
    graph = coloured_edges(genome)
    graph = two_break_on_graph(graph, a, b, c, d)
    cycles = graph_to_cycles(graph)
    genome = []
    for cycle in cycles:
        genome.append(cycle_to_genome(cycle))
    return genome


def break_distance(edges):
    graph = graph_to_cycles(edges, breakpoint_graph=True)
    distance = len(edges) / 2 - len(graph)

    return distance
