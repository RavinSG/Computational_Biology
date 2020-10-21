from collections import defaultdict

from params import *
from dna_replication import hamming_distance


def list_to_dict(adjacency_list, weighted=True):
    node_dict = defaultdict(dict)
    directed = defaultdict(dict)
    if weighted:
        for edge in adjacency_list:
            edge = edge.split('->')
            parent = edge[0]
            child, weight = edge[1].split(':')

            node_dict[parent][child] = int(weight)
            directed[parent][child] = int(weight)
            node_dict[child][parent] = int(weight)
    else:
        for edge in adjacency_list:
            edge = edge.split('->')

            node_dict[edge[0]][edge[1]] = 0
            directed[edge[0]][edge[1]] = 0
            node_dict[edge[1]][edge[0]] = 0

    return node_dict, directed


def calculate_leaf_distance(adjacency_list):
    node_dict = defaultdict(dict)
    for edge in adjacency_list:
        parent = edge[0]
        child, weight = edge[1].split(':')

        node_dict[parent][child] = int(weight)

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
    np.fill_diagonal(distances, max_val)

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


def neighbour_join(distances, nodes, k):
    n = len(distances)
    if n == 2:
        dist = distances.max()
        return {nodes[0]: {nodes[1]: dist}, nodes[1]: {nodes[0]: dist}}
    else:
        total_distance = distances.sum(axis=1)
        d_star = ((n - 2) * distances - total_distance) - total_distance.reshape(-1, 1)
        max_val = max(d_star.max() + 1, 0)
        np.fill_diagonal(d_star, max_val)
        i, j = np.unravel_index(np.argmin(d_star), d_star.shape)
        node_i = nodes[i]
        node_j = nodes[j]
        nodes.remove(node_i)
        nodes.remove(node_j)
        d_min = distances[i][j]
        delta = (total_distance[i] - total_distance[j]) / (n - 2)

        limb_i = (d_min + delta) / 2
        limb_j = (d_min - delta) / 2

        distance_k = np.delete((distances[:, j] + distances[:, i] - d_min) / 2, [i, j])
        distances = np.delete(distances, [i, j], axis=0)
        distances = np.delete(distances, [i, j], axis=1)

        distances = np.hstack((distances, distance_k.reshape(-1, 1)))
        distances = np.vstack((distances, np.append(distance_k, max_val).reshape(1, -1)))
        nodes.append(k + 1)
        tree = neighbour_join(distances, nodes, k + 1)
        for x, y in zip([node_i, node_j], [limb_i, limb_j]):
            tree[k + 1][x] = y
            tree[x] = {k + 1: y}

        return tree


def simple_parsimony(graph, char_num):
    tag = dict()
    ripe_nodes = []
    for node in graph:
        tag[node] = [-1, np.zeros(4)]
        if not node.isdigit():
            s_k = np.ones(4) * np.inf
            character = node[char_num]
            s_k[STR_TO_NUM[character]] = 0

            tag[node] = [0, s_k]
            ripe_nodes.append(list(graph[node])[0])
    ripe_nodes = list(set(ripe_nodes))
    root = None
    try:
        while ripe_nodes:
            alpha = np.ones((4, 4))
            np.fill_diagonal(alpha, 0)
            node = ripe_nodes.pop(0)
            neighbours = graph[node]

            children = [child for child in neighbours if tag[child][0] != -1]
            son, daughter = [tag[child][1] for child in children]

            son = np.min(alpha + son, axis=1)
            daughter = np.min(alpha + daughter, axis=1)

            tag[node][1] = son + daughter
            tag[node][0] = 0
            # print(node, son + daughter)
            child_n = [np.argmin(son), np.argmin(daughter)]
            for child, num in zip(children, child_n):
                tag[child][0] = 1

            parent = [i for i in graph[node] if tag[i][0] == -1][0]

            if len([i for i in graph[parent] if tag[i][0] != -1]) == 1:
                ripe_nodes.append(parent)

    except IndexError:
        tag[node][0] = NUM_TO_STR[np.argmin(son + daughter)]
        root = node
    return tag, root
