from copy import deepcopy
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


def simple_parsimony(graph, node, char_num, s_k_values):
    # graph should be in the format of {parent_1:{child_1,child_2}, parent_2:{child_3,child_4} ,...}
    if not node.isdigit():
        s_k = np.ones(4) * np.inf
        s_k[STR_TO_NUM[node[char_num]]] = 0
        s_k_values[node] = s_k
        return s_k, s_k_values

    else:
        s_k = 0
        for child in graph[node]:
            child_sk, s_k_values = simple_parsimony(graph, child, char_num, s_k_values)
            alpha = np.ones((4, 4))
            np.fill_diagonal(alpha, 0)
            s_k += (child_sk + alpha).min(axis=1)
        s_k_values[node] = s_k
        return s_k, s_k_values


def backtrack_parsimony(graph, s_k_values, node):
    parent_c = s_k_values[node]
    if node.isdigit():
        for child in graph[node]:
            child_min = s_k_values[child].min()
            if s_k_values[child][STR_TO_NUM[parent_c]] == child_min:
                s_k_values[child] = parent_c
            else:
                s_k_values[child] = NUM_TO_STR[np.argmin(s_k_values[child])]

            s_k_values = backtrack_parsimony(graph, s_k_values, child)

    return s_k_values


def create_character_sequence(graph, root, leaf_len):
    trees = []
    for i in range(leaf_len):
        trees.append(simple_parsimony(graph, root, i, {})[1])

    strings = defaultdict(str)
    score = 0
    for tree in trees:
        score += tree[root].min()
        tree[root] = NUM_TO_STR[np.argmin(tree[root])]

    trees = [backtrack_parsimony(graph, tree, root) for tree in trees]

    for node in list(trees[0]):
        for tree in trees:
            strings[node] += tree[node]

    return score, strings


def print_all_edges(graph, string_mapping, skip_root=None):
    nodes = [list(graph)[0]]
    if skip_root is not None:
        nodes = [x for x in graph[skip_root]]
        node_1 = string_mapping[nodes[0]]
        node_2 = string_mapping[nodes[1]]
        dst = hamming_distance(node_1, node_2)
        print(f"{node_1}->{node_2}:{dst}")
        print(f"{node_2}->{node_1}:{dst}")

    while nodes:
        node = nodes.pop(0)
        parent = string_mapping[node]
        for child in graph[node]:
            nodes.append(child)
            child = string_mapping[child]
            dst = hamming_distance(parent, child)
            print(f"{parent}->{child}:{dst}")
            print(f"{child}->{parent}:{dst}")


def insert_first_root(adjacency_list):
    new_root = str(len(adjacency_list) // 2 + 1)
    old_edge = adjacency_list[0]
    node_1, node_2 = old_edge.split('->')
    adjacency_list.remove(old_edge)
    adjacency_list.remove('->'.join([node_2, node_1]))
    adjacency_list += ['->'.join([new_root, node]) for node in [node_1, node_2]]
    adjacency_list += ['->'.join([node, new_root]) for node in [node_1, node_2]]

    return adjacency_list, new_root


def directed_binary_tree(adjacency_list, root=None):
    # converts an undirected binary to a directed tree starting fro the node
    tree, _ = list_to_dict(adjacency_list, False)
    if root is None:
        for key, value in tree.items():
            if len(value) == 2:
                root = key
                break
    visited = []
    cur_nodes = [root]
    parent = defaultdict(dict)
    while cur_nodes:
        node = cur_nodes.pop(0)
        for child in tree[node]:
            if child in visited:
                continue
            else:
                parent[node][child] = 0
                cur_nodes.append(child)
        visited.append(node)

    return parent, root


def move_root(parent_graph, old_root, leaf_len):
    score, str_mapping = create_character_sequence(parent_graph, old_root, leaf_len)
    left_node = list(parent_graph[old_root])[0]
    right_node = list(parent_graph[old_root])[1]

    parent_graph.pop(old_root)
    left_parents = deepcopy(parent_graph)
    right_parents = deepcopy(parent_graph)
    dirs = ['left', 'right']

    for child, direction in zip(parent_graph[left_node], dirs):
        score = min(score, move_sub_tree(left_parents, old_root, child, left_node, right_node, direction, leaf_len))
    for child, direction in zip(parent_graph[right_node], dirs):
        score = min(score, move_sub_tree(right_parents, old_root, child, right_node, left_node, direction, leaf_len))

    return score


def move_sub_tree(parents, root, start_node, parent_node, sibling, direction, leaf_len):
    if start_node in parents:
        l_child, r_child = parents[start_node]
        if direction == 'left':
            left_parents = deepcopy(parents)
            left_parents[parent_node].pop(start_node)
            left_parents[root][start_node] = 0
            left_parents[root][parent_node] = 0
            left_parents[parent_node] = {sibling: 0, list(left_parents[parent_node])[0]: 0}
            score, str_mapping = create_character_sequence(left_parents, root, leaf_len)
            left_parents.pop(root)
            s_1 = move_sub_tree(left_parents, root, l_child, start_node, parent_node, 'left', leaf_len)
            s_2 = move_sub_tree(left_parents, root, r_child, start_node, parent_node, 'right', leaf_len)
            return min(score, s_1, s_2)
        else:
            right_parents = deepcopy(parents)
            right_parents[parent_node].pop(start_node)
            right_parents[root][parent_node] = 0
            right_parents[root][start_node] = 0
            right_parents[parent_node] = {list(right_parents[parent_node])[0]: 0, sibling: 0}
            score, str_mapping = create_character_sequence(right_parents, root, leaf_len)
            right_parents.pop(root)
            s_1 = move_sub_tree(right_parents, root, r_child, start_node, parent_node, 'right', leaf_len)
            s_2 = move_sub_tree(right_parents, root, l_child, start_node, parent_node, 'left', leaf_len)
            return min(score, s_1, s_2)
    else:
        node_parents = deepcopy(parents)
        node_parents[parent_node].pop(start_node)
        node_parents[root][start_node] = 0
        node_parents[root][parent_node] = 0
        node_parents[parent_node][sibling] = 0
        score, str_mapping = create_character_sequence(node_parents, root, leaf_len)
        return score
