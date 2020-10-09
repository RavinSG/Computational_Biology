from itertools import product
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


def de_bruijn_graph(strand, k):
    if type(strand) == str:
        k_mers = composition(strand, k)
    else:
        k_mers = strand
    nodes = defaultdict(list)

    for k_mer in k_mers:
        nodes[k_mer[:-1]].append(k_mer[1:])

    return nodes


def generate_node_dict(adjacency_list):
    node_dict = defaultdict(list)
    total_edges = 1
    for i in adjacency_list:
        nodes = i.split(" -> ")
        dst_nodes = list(map(str, nodes[1].split(',')))
        total_edges += len(dst_nodes)
        node_dict[nodes[0]] = dst_nodes

    return node_dict


def unbalanced_nodes(adjacency_list, debruijn=False):
    if not debruijn:
        node_dict = generate_node_dict(adjacency_list)
    else:
        node_dict = adjacency_list

    in_edges = defaultdict(int)

    for nodes in node_dict.values():
        for node in nodes:
            in_edges[node] += 1

    total_nodes = set(list(node_dict.keys()) + list(in_edges.keys()))
    missing_edge = [None, None]
    unbalance_nodes = [[node, in_edges[node], len(node_dict[node])] for node in total_nodes if
                       in_edges[node] != len(node_dict[node])]

    for node in unbalance_nodes:
        if node[1] < node[2]:
            missing_edge[1] = node[0]
        else:
            missing_edge[0] = node[0]

    node_dict[missing_edge[0]].append(missing_edge[1])
    return node_dict, missing_edge


def generate_eulerian_walk(node_dict):
    walk = []
    next_node = list(node_dict)[0]
    while True:
        try:
            walk.append(next_node)
            next_node = node_dict[next_node].pop()

        except IndexError:
            for i, j in enumerate(walk):
                if node_dict[j]:
                    walk = walk[i:] + walk[1:i]
                    next_node = j
                    break
                else:
                    return walk


def generate_eulerian_cycle(node_dict):
    eulerian_cycle = generate_eulerian_walk(node_dict)

    return eulerian_cycle


def generate_eulerian_path(node_dict, extra_edge):
    eulerian_path = generate_eulerian_walk(node_dict)

    break_point = [x for x, y in enumerate(eulerian_path) if
                   (y == extra_edge[0] and eulerian_path[x + 1] == extra_edge[1])][0]

    eulerian_path = eulerian_path[break_point + 1:-1] + eulerian_path[:break_point + 1]
    return eulerian_path


def eulerian_walk(adjacency_list, eulerian_path=False):
    if eulerian_path:
        node_dict, extra_edge = unbalanced_nodes(adjacency_list)
        return generate_eulerian_path(node_dict, extra_edge)
    else:
        node_dict = generate_node_dict(adjacency_list)
        return generate_eulerian_cycle(node_dict)


def string_reconstruction(reads):
    reads = de_bruijn_graph(reads, 0)
    node_dict, extra_edge = unbalanced_nodes(reads, debruijn=True)
    return simple_string_assembly(generate_eulerian_path(node_dict, extra_edge))


def k_universal_string(k):
    strings = ["".join(x) for x in product("01", repeat=k)]
    strings = de_bruijn_graph(strings, 0)
    cycle = generate_eulerian_walk(strings)
    print(len(cycle), cycle)
    return simple_string_assembly(cycle[:-(k - 1)])
