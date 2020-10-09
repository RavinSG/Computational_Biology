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


def generate_eulerian_cycle(adjacency_list):
    node_dict = dict()
    total_edges = 1
    for i in adjacency_list:
        nodes = i.split(" -> ")
        dst_nodes = list(map(int, nodes[1].split(',')))
        total_edges += len(dst_nodes)
        node_dict[int(nodes[0])] = dst_nodes

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
                return eulerian_cycle
