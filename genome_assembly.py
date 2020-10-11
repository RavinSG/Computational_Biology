from copy import deepcopy
from collections import defaultdict
from itertools import permutations, product


def composition(strand: str, k: int) -> list:
    """
    Breaks the strand into overlapping k_mers.
    """
    reads = [strand[x:x + k] for x in range(len(strand) - k + 1)]
    reads.sort()
    return reads


def simple_string_assembly(strands: list) -> str:
    """
    Joins compositions and create the original string.
    """
    start = strands[0]
    start = start + ''.join(x[-1] for x in strands[1:])
    return start


def simple_overlapping_graph(strands: list) -> defaultdict:
    """
    Given a set of strands, creates a directed graph where there is an edge from node a to b if suffix(a) = prefix(b).

    :param strands: A list of k_mers
    :return: The adjacency list of the graph
    """
    prefixes = defaultdict(list)
    for i in strands:
        prefixes[i[:-1]].append(i)

    adjacency_list = defaultdict(list)
    for i in strands:
        strings = prefixes[i[1:]]
        if len(strings) > 0:
            adjacency_list[i] = strings
            print(i, '->', ",".join(strings))

    return adjacency_list


def de_bruijn_graph(k_mers):
    """
    Same as simple_overlapping_graph without the print.
    """
    nodes = defaultdict(list)

    for k_mer in k_mers:
        nodes[k_mer[:-1]].append(k_mer[1:])

    return nodes


def generate_node_dict(adjacency_list: list) -> defaultdict:
    """
    Given an adjacency list in the following format, returns a dictionary of nodes with the adjacent nodes as values.
    [[node_1 -> node_2,node_3],
    [node_2 -> node_4],
    ....]

    :param adjacency_list: A list of adjacency
    :return: A dictionary of nodes
    """
    node_dict = defaultdict(list)
    total_edges = 1
    for i in adjacency_list:
        nodes = i.split(" -> ")
        dst_nodes = list(map(str, nodes[1].split(',')))
        total_edges += len(dst_nodes)
        node_dict[nodes[0]] = dst_nodes

    return node_dict


def calculate_edges(node_dict: dict) -> tuple:
    """
    Given a node dictionary of a directed graph calculates the in degree and out degree of the nodes.

    :param node_dict: An adjacency list in the dictionary format
    :return: A tuple of two dictionaries with the nodes as keys
    """
    in_degree = defaultdict(int)
    out_degree = defaultdict(int)

    for s_node, nodes in node_dict.items():
        out_degree[s_node] = len(nodes)
        for node in nodes:
            in_degree[node] += 1

    return in_degree, out_degree


def unbalanced_nodes(adjacency_list, dict_form=False) -> tuple:
    """
    Given a adjacency list of a graph with two unbalanced nodes, adds a node between the two nodes in the adjacency
    list, and returns the added edge.

    :param adjacency_list: An adjacency list in either dictionary or list format.
    :param dict_form: True if in dictionary format
    :return: The updates adjacency list and the extra edge
    """
    if not dict_form:
        node_dict = generate_node_dict(adjacency_list)
    else:
        node_dict = adjacency_list

    in_degree, out_degree = calculate_edges(node_dict)

    total_nodes = set(list(node_dict.keys()) + list(in_degree.keys()))
    missing_edge = [None, None]
    # If the in-degree is not equal to the out-degree add the node to the list with the number of degree.
    unbalance_nodes = [[node, in_degree[node], out_degree[node]] for node in total_nodes if
                       in_degree[node] != len(node_dict[node])]

    for node in unbalance_nodes:
        if node[1] < node[2]:
            missing_edge[1] = node[0]
        else:
            missing_edge[0] = node[0]

    node_dict[missing_edge[0]].append(missing_edge[1])
    return node_dict, missing_edge


def generate_eulerian_walk(node_dict: dict) -> list:
    """
    Given a graph G's node dictionary, returns an eulerian walk/cycle in G. Note that this function assumes that there
    is an eulerian cycle present in the graph.

    :param node_dict: An adjacency list in the dictionary format
    :return: The list of nodes in the walk
    """
    walk = []
    next_node = list(node_dict)[0]
    while True:
        try:
            walk.append(next_node)
            next_node = node_dict[next_node].pop()

        except IndexError:
            for i, j in enumerate(walk):
                if node_dict[j]:
                    # If there is a node in the walk that has unvisited adjacent nodes, start a new walk from the node
                    walk = walk[i:] + walk[1:i]
                    next_node = j
                    break
                else:
                    return walk


def generate_eulerian_cycle(node_dict: dict) -> list:
    """
    Same as eulerian walk.
    """
    eulerian_cycle = generate_eulerian_walk(node_dict)

    return eulerian_cycle


def generate_eulerian_path(node_dict: dict, extra_edge: list) -> list:
    """
    Given a graph G's node dictionary, returns an eulerian path in G. Note that this function assumes that there is
    an eulerian path present in the graph.

    :param node_dict: An adjacency list in the dictionary format
    :param extra_edge: The additional edge from the end node to start node
    :return: The list of nodes in the walk
    """
    eulerian_path = generate_eulerian_walk(node_dict)

    # Find the additional edge and break it to create the path
    break_point = [x for x, y in enumerate(eulerian_path) if
                   (y == extra_edge[0] and eulerian_path[x + 1] == extra_edge[1])][0]

    eulerian_path = eulerian_path[break_point + 1:-1] + eulerian_path[:break_point + 1]
    return eulerian_path


def eulerian_walk(adjacency_list, eulerian_path=False, dict_form=False) -> list:
    """
    A common interface for creating eulerian cycles and eulerian paths from a graph.

    :param adjacency_list: A list of adjacency
    :param eulerian_path: If true returns a path instead of a cycle
    :param dict_form: True of the adjacency list in dict format
    :return: The list of nodes in the eulerian walk
    """
    if eulerian_path:
        node_dict, extra_edge = unbalanced_nodes(adjacency_list, dict_form)
        return generate_eulerian_path(node_dict, extra_edge)
    else:
        if dict_form:
            node_dict = adjacency_list
        else:
            node_dict = generate_node_dict(adjacency_list)
        return generate_eulerian_cycle(node_dict)


def string_reconstruction(reads: list) -> str:
    """
    Given a list of reads, using the de Bruijn graph the underlying string is reconstructed.

    :param reads: A list of overlapping reads
    :return: The underlying string
    """
    reads = de_bruijn_graph(reads)
    node_dict, extra_edge = unbalanced_nodes(reads, dict_form=True)
    return simple_string_assembly(generate_eulerian_path(node_dict, extra_edge))


def k_universal_string(k: int) -> str:
    """
    Generates a universal string of length k from the alphabet '01'
    """
    strings = ["".join(x) for x in product("01", repeat=k)]
    strings = de_bruijn_graph(strings)
    cycle = generate_eulerian_walk(strings)

    return simple_string_assembly(cycle[:-(k - 1)])


def k_d_mers(strand: str, k: int, d: int) -> list:
    """
    Given a dna strand, find pairs of k_mers that are d distance apart measured from the ending location of the first
    k_mer and the starting location of the second k_mer.

    :param strand: A dna strand
    :param k: Length of the k_mer
    :param d: Distance between the pair of k_mers
    :return: A list containing all paired k_mers
    """
    k_mer_pairs = []
    for i in range(len(strand) - (2 * k + d) + 1):
        k_mer_1 = strand[i:i + k]
        k_mer_2 = strand[i + k + d:i + d + 2 * k]
        k_mer_pairs.append(f"({k_mer_1}|{k_mer_2})")

    k_mer_pairs.sort()
    return k_mer_pairs


def string_spelled_by_gapped_patterns(gapped_patterns: list, k: int, d: int) -> str:
    """
    Given a list of organized paired reads of length k, d distance apart, constructs the underlying string.

    The paired reads should be in the format of: k_mer1|kmer_2

    :param gapped_patterns: A list of paired reads
    :param k: Length of a read
    :param d: Pair distance
    :return: Underlying string
    """
    prefix = []
    suffix = []
    for i in gapped_patterns:
        string = i.split("|")
        prefix.append(string[0])
        suffix.append(string[1])

    prefix = simple_string_assembly(prefix)
    suffix = simple_string_assembly(suffix)

    for i in range(k + d, len(prefix)):
        # Check if the two strings constructed from the reads overlap in the correct positions
        if prefix[i] != suffix[i - k - d]:
            return "Not Found"
    else:
        return prefix + suffix[-(k + d):]


def read_pair_string_construction(reads: list, k: int, d: int) -> str:
    """
    Given a list of reads, first create a de bruijn graph from them. Then tries to find an eulerian walk in the created
    graph. If a walk is found using string_spelled_by_gapped_patterns creates a string that matches the walk.

    The paired reads should be in the format of: k_mer1|kmer_2

    :param reads: A list of paired reads
    :param k: Length of k_mers
    :param d: Distance between read pairs
    :return: The underlying string
    """
    node_dict = defaultdict(list)
    # Break the paired reads and create nodes
    for i in reads:
        edge = i.split("|")
        node_dict[edge[0][:-1] + "|" + edge[1][:-1]].append(edge[0][1:] + "-" + edge[1][1:])

    node_dict, missing_edge = unbalanced_nodes(node_dict, True)
    # Find nodes with more than one out-going edge since they lead to multiple cycles
    split_points = {x: len(y) for x, y in node_dict.items() if len(y) > 1}
    for i in split_points:
        split_points[i] = list(permutations(node_dict[i]))

    # Create a list of all possible paths in the graph and iterate through them till a valid string is constructed
    all_paths = list(set(product(*list(split_points.values()))))

    # It would be much efficient to traceback and create select a new path, but too bored to do it.
    while all_paths:
        cur_path = all_paths.pop()
        temp_dict = deepcopy(node_dict)
        for i, j in enumerate(split_points.keys()):
            temp_dict[j] = list(cur_path[i])
        string = generate_eulerian_path(temp_dict, missing_edge)

        constructed_string = string_spelled_by_gapped_patterns(string, k, d)
        if constructed_string != "Not Found":
            return constructed_string


def maximal_non_branching_paths(nodes: dict) -> list:
    """
    Given a node dictionary of a graph, find all path whose internal nodes are 1-in-1-out nodes and whose initial and
    final nodes are not 1-in-1-out nodes. All paths in isolated cycles are also returned.

    A node is a 1-in-1-out node if its in-degree and out-degree are both equal to 1

    :param nodes: A node dictionary
    :return: A list of all non branching paths
    """
    in_edges, out_edges = calculate_edges(nodes)
    one_in_out_nodes = defaultdict(bool)

    # Find all one-in-one-out nodes in the graph
    for node in nodes.keys():
        one_in_out_nodes[node] = True if (in_edges[node] * out_edges[node] == 1) else False

    added_nodes = set()
    non_branching_paths = []
    items = list(zip(*nodes.items()))

    for node, paths in zip(items[0], items[1]):
        if not one_in_out_nodes[node]:
            # For each node that is adjacent to the node create paths
            for path in paths:
                non_branching_path = [node, path]
                added_nodes.update(non_branching_path)
                while True:
                    if len(nodes[path]) > 0 and one_in_out_nodes[path]:
                        path = nodes[path][0]
                        non_branching_path.append(path)
                        added_nodes.add(path)
                    else:
                        non_branching_paths.append(non_branching_path)
                        break

    for s_node, check in one_in_out_nodes.items():
        # Since isolated cycles are not reachable from the starting node of the graph, search for nodes that are not
        # present in the created paths.
        if check and (s_node not in added_nodes):
            path = nodes[s_node][0]
            non_branching_path = [s_node, path]
            added_nodes.update([s_node, path])
            # Visit adjacent node till the starting node is reached
            while s_node != path:
                path = nodes[path][0]
                added_nodes.add(path)
                non_branching_path.append(path)
            non_branching_paths.append(non_branching_path)

    return [simple_string_assembly(i) for i in non_branching_paths]
