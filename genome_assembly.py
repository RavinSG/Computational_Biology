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

    for i, j in nodes.items():
        print(i, "->", ",".join(j))


def generate_node_dict(adjacency_list):
    node_dict = defaultdict(list)
    total_edges = 1
    for i in adjacency_list:
        nodes = i.split(" -> ")
        dst_nodes = list(map(int, nodes[1].split(',')))
        total_edges += len(dst_nodes)
        node_dict[int(nodes[0])] = dst_nodes

    return node_dict, total_edges


def unbalanced_nodes(adjacency_list):
    node_dict, total_edges = generate_node_dict(adjacency_list)
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
    return node_dict, total_edges + 1, missing_edge


def generate_eulerian_cycle(adjacency_list, eulerian_path=False):
    if eulerian_path:
        node_dict, total_edges, extra_edge = unbalanced_nodes(adjacency_list)
    else:
        node_dict, total_edges = generate_node_dict(adjacency_list)

    eulerian_cycle = []
    next_node = list(node_dict)[0]

    while True:
        try:
            eulerian_cycle.append(next_node)
            next_node = node_dict[next_node].pop()

        except IndexError:
            for i, j in enumerate(eulerian_cycle):
                if node_dict[j]:
                    eulerian_cycle = eulerian_cycle[i:] + eulerian_cycle[1:i]
                    next_node = j
                    break
            else:
                if eulerian_path:
                    break_point = [x for x, y in enumerate(eulerian_cycle) if
                                   (y == extra_edge[0] and eulerian_cycle[x + 1] == extra_edge[1])][0]
                    eulerian_cycle = eulerian_cycle[break_point + 1:-1] + eulerian_cycle[:break_point + 1]
                    return eulerian_cycle
                else:
                    return eulerian_cycle
