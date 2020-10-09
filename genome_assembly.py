from itertools import permutations, product
from collections import defaultdict
from copy import deepcopy


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


def calculate_edges(node_dict):
    in_edges = defaultdict(int)
    out_edges = defaultdict(int)

    for s_node, nodes in node_dict.items():
        out_edges[s_node] = len(nodes)
        for node in nodes:
            in_edges[node] += 1

    return in_edges, out_edges


def unbalanced_nodes(adjacency_list, dict_form=False):
    if not dict_form:
        node_dict = generate_node_dict(adjacency_list)
    else:
        node_dict = adjacency_list

    in_edges, _ = calculate_edges(node_dict)

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
    node_dict, extra_edge = unbalanced_nodes(reads, dict_form=True)
    return simple_string_assembly(generate_eulerian_path(node_dict, extra_edge))


def k_universal_string(k):
    strings = ["".join(x) for x in product("01", repeat=k)]
    strings = de_bruijn_graph(strings, 0)
    cycle = generate_eulerian_walk(strings)
    print(len(cycle), cycle)
    return simple_string_assembly(cycle[:-(k - 1)])


def k_d_mers(strand, k, d):
    k_mer_pairs = []
    for i in range(len(strand) - (2 * k + d) + 1):
        k_mer_1 = strand[i:i + k]
        k_mer_2 = strand[i + k + d:i + d + 2 * k]
        k_mer_pairs.append(f"({k_mer_1}|{k_mer_2})")

    k_mer_pairs.sort()
    print(" ".join(k_mer_pairs))


def string_spelled_by_gapped_patterns(gapped_patterns, k, d):
    prefix = []
    suffix = []
    for i in gapped_patterns:
        string = i.split("|")
        prefix.append(string[0])
        suffix.append(string[1])
    prefix = simple_string_assembly(prefix)
    suffix = simple_string_assembly(suffix)
    print(prefix)
    str_len = len(gapped_patterns) + 2 * k + d - 1
    print(str_len, k + d, len(prefix))
    for i in range(k + d, len(prefix)):
        if prefix[i] != suffix[i - k - d]:
            return "Not Found"
    else:
        return prefix + suffix[-(k + d):]


def read_pair_string_construction(reads, k, d):
    node_dict = defaultdict(list)
    for i in reads:
        edge = i.split("|")
        node_dict[edge[0][:-1] + "-" + edge[1][:-1]].append(edge[0][1:] + "-" + edge[1][1:])

    node_dict, missing_edge = unbalanced_nodes(node_dict, True)
    split_points = {x: len(y) for x, y in node_dict.items() if len(y) > 1}
    for i in split_points:
        split_points[i] = list(permutations(node_dict[i]))

    all_paths = list(set(product(*list(split_points.values()))))

    # It would be much efficient to traceback and create select a new path, but too bored to do it.
    while all_paths:
        cur_path = all_paths.pop()
        temp_dict = deepcopy(node_dict)
        for i, j in enumerate(split_points.keys()):
            temp_dict[j] = list(cur_path[i])
        string = generate_eulerian_path(temp_dict, missing_edge)
        prefix = []
        suffix = []
        for i in string:
            splices = i.split('-')
            prefix.append(splices[0])
            suffix.append(splices[1])

        prefix = simple_string_assembly(prefix)
        suffix = simple_string_assembly(suffix)
        for i in range(k + d, len(prefix)):
            if prefix[i] != suffix[i - k - d]:
                print("Not Found")
                break
        else:
            return prefix + suffix[-(k + d):]


def maximum_non_branching_paths(nodes):
    in_edges, out_edges = calculate_edges(nodes)
    one_in_out_nodes = defaultdict(bool)

    for node in nodes.keys():
        one_in_out_nodes[node] = True if (in_edges[node] * out_edges[node] == 1) else False
    for i in nodes.keys():
        if i not in one_in_out_nodes.keys():
            print(i, "*" * 100)
    added_nodes = set()
    non_branching_paths = []
    items = list(zip(*nodes.items()))

    for node, paths in zip(items[0], items[1]):
        if not one_in_out_nodes[node]:
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
        if check and (s_node not in added_nodes):
            path = nodes[s_node][0]
            non_branching_path = [s_node, path]
            added_nodes.update([s_node, path])
            while s_node != path:
                path = nodes[path][0]
                added_nodes.add(path)
                non_branching_path.append(path)
            non_branching_paths.append(non_branching_path)

    # return non_branching_paths
    return [simple_string_assembly(i) for i in non_branching_paths]
