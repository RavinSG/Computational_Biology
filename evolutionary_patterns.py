import numpy as np
from collections import defaultdict


def calculate_leaf_distance(adjacency_list):
    node_dict = defaultdict(dict)
    for edge in adjacency_list:
        parent = edge[0]
        child, weight = edge[1].split(':')

        node_dict[int(parent)][int(child)] = int(weight)

    num_nodes = len(node_dict)
    distances = np.ones((num_nodes, num_nodes)) * np.inf

    for i in node_dict:
        distances[i][i] = 0
        for j, k in node_dict[i].items():
            distances[i][j] = k

    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if distances[i][j] > distances[i][k] + distances[k][j]:
                    distances[i][j] = distances[i][k] + distances[k][j]

    leaves = [x for x, y in node_dict.items() if len(y) == 1]
    return distances[leaves][:, leaves]


def limb_length(distances, leaf):
    num_nodes = len(distances)
    length = np.inf

    for i in range(min(len(distances), 5)):
        for j in range(num_nodes):
            if i == leaf or j == leaf:
                continue
            dist = (distances[i][leaf] + distances[leaf][j] - distances[i][j]) / 2
            if dist < length:
                length = dist

    return length


def find_path_nodes(matrix, n, limb_len):
    for i in range(n):
        for j in range(n):
            if matrix[i][j] == matrix[i][n] + matrix[n][j] - limb_len * 2:
                return i, j

    return -1


def get_path(node_1, node_2, graph):
    visited = []
    stack = [node_1]
    parents = {}
    while stack:
        node = stack.pop()
        for child in graph[node]:
            if child not in visited:
                parents[child] = node
            else:
                continue
            if child != node_2:
                stack.append(child)
            else:
                path = [child]
                while child != node_1:
                    child = parents[child]
                    path.append(child)
                path.reverse()
                return path
        visited.append(node)


def additive_phylogeny(matrix, num_nodes):
    n = len(matrix) - 1
    if n == 1:
        return {0: {1: matrix[0][1]}, 1: {0: matrix[1][0]}}, num_nodes
    else:
        limb_len = limb_length(matrix, n)

        tree, num_nodes = additive_phylogeny(matrix[:-1, :][:, :-1], num_nodes)

        i, j = find_path_nodes(matrix, n, limb_len)
        path = get_path(i, j, tree)
        dist = 0
        break_dist = matrix[i, n] - limb_len
        for p in range(len(path) - 1):
            cur_node = path[p]
            next_node = path[p + 1]
            edge_distance = tree[cur_node][next_node]
            dist += edge_distance
            if dist > break_dist:
                tree[cur_node].pop(next_node)
                tree[next_node].pop(cur_node)

                tree[num_nodes] = {cur_node: break_dist - (dist - edge_distance), next_node: dist - break_dist,
                                   n: limb_len}
                tree[n] = {num_nodes: limb_len}
                tree[cur_node][num_nodes] = break_dist - (dist - edge_distance)
                tree[next_node][num_nodes] = dist - break_dist
                num_nodes += 1
                break
            elif dist == break_dist:
                tree[next_node][n] = limb_len
                tree[n] = {next_node: limb_len}
                break

        return tree, num_nodes


def upgma(distances, n):
    clusters = [(x,) for x in range(n)]
    graph = {(x,): [0] for x in range(n)}
    max_val = distances.max() + 1
    for i in range(n):
        distances[i][i] = max_val

    while len(clusters) > 1:
        i, j = np.unravel_index(np.argmin(distances), distances.shape)
        c_i = len(clusters[i])
        c_j = len(clusters[j])
        d_i_j = distances[i, j]

        col = (c_i * distances[:, i] + c_j * distances[:, j]) / (c_i + c_j)
        distances = np.delete(distances, [i, j], axis=0)
        distances = np.delete(distances, [i, j], axis=1)

        new_cluster = clusters[i] + clusters[j]
        graph[new_cluster] = [d_i_j / 2]
        graph[clusters[i]] = [graph[clusters[i]][0], new_cluster]
        graph[clusters[j]] = [graph[clusters[j]][0], new_cluster]

        if i > j:
            clusters = clusters[:j] + clusters[j + 1:i] + clusters[i + 1:]
            col = np.concatenate((col[:j], col[j + 1:i], col[i + 1:]))
        else:
            clusters = clusters[:i] + clusters[i + 1:j] + clusters[j + 1:]
            col = np.concatenate((col[:i], col[i + 1:j], col[j + 1:]))

        distances = np.hstack((distances, col.reshape(-1, 1)))
        distances = np.vstack((distances, np.append(col, max_val).reshape(1, -1)))
        clusters.append(new_cluster)
        n = n + 1

    return graph


def print_edge_distances(graph, leaf_count):
    mapping = dict()
    for i in graph:
        if len(i) == 1:
            mapping[i] = i[0]
        else:
            mapping[i] = leaf_count
            leaf_count += 1

    age_graph = defaultdict(dict)
    try:
        for key, value in graph.items():
            age_graph[mapping[key]][mapping[value[1]]] = graph[value[1]][0] - value[0]
            age_graph[mapping[value[1]]][mapping[key]] = graph[value[1]][0] - value[0]
    except IndexError:
        pass

    for i in range(leaf_count):
        for key, value in age_graph[i].items():
            print(f"{i}->{key}:{value}")
