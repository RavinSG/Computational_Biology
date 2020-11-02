import numpy as np
from collections import defaultdict

from dna_replication import get_complement

"""
(+1, -2, -3, +4)        - genome
(2, 3), (4, 5), (6, 1)  - coloured edges/ graph
(1, 2, 3, 4, 5, 6)      - node sequence/ cycle
"""


def greedy_sort(perm: list) -> tuple:
    """
    Solves the reversal problem by aligning each sorting the permutation in ascending order.
    (+1 −7 +6 −10 +9 −8 +2 -11 -3 +5 +4)
    (+1 -2 +8 -9 +10 -6 +7 -11 -3 +5 +4)
    (+1 +2 +8 -9 +10 -6 +7 -11 -3 +5 +4)
    ...

    :param perm: A permutation of synteny alignments
    :return: The aligned syntenies with the number of steps taken
    """
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


def count_breakpoints(perm: list) -> int:
    """
    Given a synteny alignment calculates the number of breakpoints in the alignment.

    :param perm: A permutation of synteny alignments
    :return: Number of breakpoints
    """
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


def chromosome_to_cycle(chromosome: list) -> list:
    """
    Transform a single circular chromosome, Chromosome = (Chromosome_1, . . . , Chromosome_n) into a cycle represented
    as a sequence of integers Nodes = (Nodes_1, . . . , Nodes_2n).

    :param chromosome: A circular chromosome
    :return: The node sequence of the cycle
    """
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


def cycle_to_chromosome(cycle: list) -> list:
    """
    Inverts the chromosome_to_cycle() and returns the original circular chromosome.

    :param cycle: A node sequence
    :return: The circular chromosome responsible for the cycle
    """
    chromosome = []
    k = len(cycle)
    for i in range(0, k, 2):
        head, tail = cycle[i:i + 2]
        if head < tail:
            chromosome.append(tail // 2)
        else:
            chromosome.append(-head // 2)
    return chromosome


def coloured_edges(chromosomes: list) -> list:
    """
    Colored edges are defined as edges joining synteny blocks in a chromosome.

    :param chromosomes: A list of chromosomes
    :return: The coloured edges of the graph created by the choromosomes
    """
    edges = []
    for chromosome in chromosomes:
        cycle = chromosome_to_cycle(chromosome)
        k = len(cycle)
        for i in range(1, k - 2, 2):
            edges.append((cycle[i], cycle[i + 1]))

        edges.append((cycle[-1], cycle[0]))

    return edges


def graph_to_cycles(graph: list, breakpoint_graph=False) -> list:
    """
    Given the coloured edges of a graph, creates the circular chromosomes that are responsible for the coloured edges.
    If the graph is in the form of a breakpoint graph the number of cycles reflects the break distance between the two
    chromosomes.

    :param graph: An edge list
    :param breakpoint_graph: If true the breakpoint graph is generated
    :return: The cycles present in the graph
    """
    nodes = defaultdict(list)

    for i in graph:
        nodes[i[0]].append(i[1])
        nodes[i[1]].append(i[0])

    # If the graph is the breakpoint graph, two ends of the synteny block is connected to each other
    if not breakpoint_graph:
        for i in nodes.keys():
            nodes[i].append(i + 1 if i % 2 == 1 else i - 1)

    chromosomes = []
    prev_node = list(nodes.keys())[0]
    next_node = nodes[prev_node][0]
    # Remove the edge by removing the ending and starting nodes from the possible transitions in respective nodes
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
        # When the starting node is reached there will be no outgoing edges
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


def two_break_on_graph(graph: list, a: int, b: int, c: int, d: int) -> list:
    """
    The two-break(a, b, c, d) is defined as replaces colored edges (a, b) and (c, d) in a genome graph with two new
    colored edges (a, c) and (b, d).

    :param graph: The graph the two break is performed on
    :param a: End_1 of edge_1
    :param b: End_2 of edge_1
    :param c: End_1 of edge_2
    :param d: End_2 of edge_2
    :return: The updated graph
    """
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
    """
    Same as two_break_on_graph, but defined on the genome.

    :param genome: The genome the two break should be carried out on
    :param a: End_1 of edge_1
    :param b: End_2 of edge_1
    :param c: End_1 of edge_2
    :param d: End_2 of edge_2
    :return: The updated genome
    """
    graph = coloured_edges(genome)
    graph = two_break_on_graph(graph, a, b, c, d)
    cycles = graph_to_cycles(graph)
    genome = []
    for cycle in cycles:
        genome.append(cycle_to_chromosome(cycle))
    return genome


def break_distance(genomes: list) -> int:
    """
    Calculated the break distance between two genomes P and Q. The break distance between P and Q is defines as the
    length of the shortest sequence of 2-breaks transforming genome P into genome Q.

    A 2-break is defined as the removal of two edges on a genome graph and replacing them with new two edges on the same
    four nodes.

    :param genomes: A list of genomes P and Q
    :return: The break distance between the two genomes
    """
    edges = coloured_edges(genomes)
    graph = graph_to_cycles(edges, breakpoint_graph=True)
    distance = int(len(edges) / 2 - len(graph))

    return distance


def two_break_sorting(genomes: list) -> list:
    """
    Given two genomes with circular chromosomes on the same synteny blocks, generates the sequence of genomes resulting
    from applying a shortest sequence of 2-breaks transforming one genome into the other.

    :param genomes: A list of genomes P and Q
    :return: The evolution history of one genome transforming to another
    """
    edges = coloured_edges(genomes)
    graph = graph_to_cycles(edges, breakpoint_graph=True)
    genome_1 = [genomes[0]]
    genome_evolution = [genome_1]
    # for k in genome_1:
    #     print("(" + " ".join(f'{x:+d}' for x in list(map(int, k))) + ")")
    for cycle in graph:
        if len(cycle) == 2:
            continue
        else:
            for i in range(2, len(cycle), 2):
                break_points = [cycle[i - 2], cycle[-1], cycle[i - 1], cycle[i]]
                genome_1 = two_break_on_genome(genome_1, *break_points)
                genome_evolution.append(genome_1)

    return genome_evolution


def shared_k_mers(k: int, strand_1: str, strand_2: str) -> list:
    """
    A shared k-mer is defined as a k-mer shared by two genomes if either the k-mer or its reverse complement appears in
    each genome.

    :param k: Length of the k-mer
    :param strand_1: Chromosome 1
    :param strand_2: Chromosome 2
    :return: The positions of the shared k-mers
    """
    comp_2 = get_complement(strand_2, True)
    k_mer_dict = defaultdict(list)
    positions = []
    len_2 = len(strand_2)
    for i in range(len(strand_1) - k + 1):
        k_mer_dict[strand_1[i:i + k]].append(i)

    for i in range(len(strand_2) - k + 1):
        if strand_2[i:i + k] in k_mer_dict:
            for pos in k_mer_dict[strand_2[i:i + k]]:
                positions.append((pos, i))

        if comp_2[i:i + k] in k_mer_dict:
            for pos in k_mer_dict[comp_2[i:i + k]]:
                rev = len_2 - k - i
                positions.append((pos, rev))

    return positions
