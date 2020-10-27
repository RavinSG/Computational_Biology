import numpy as np
from collections import defaultdict


def farthest_first_travel(data_points, k):
    data_points = np.array(data_points)
    centers = np.array([data_points[0]])

    while len(centers) < k:
        distances = np.sum((data_points - centers.reshape((len(centers), 1, -1))) ** 2, axis=2) ** 0.5
        next_cluster = np.argmax(np.min(distances, axis=0))
        centers = np.concatenate((centers, [data_points[next_cluster]]))

    return centers


def max_distance(data_points, centers):
    return np.max(
        np.min(np.sum((data_points - np.array(centers).reshape((len(centers), 1, -1))) ** 2, axis=2), axis=0) ** 0.5)


def calculate_distortion(data_points, centers):
    data_points = np.array(data_points)
    centers = np.array(centers).reshape((len(centers), 1, -1))

    distances = np.sum((data_points - centers.reshape((len(centers), 1, -1))) ** 2, axis=2)
    return np.min(distances, axis=0).sum() / len(data_points)


def k_means_initializer(data_points: np.ndarray, k):
    centers = np.array([data_points[np.random.choice(len(data_points))]])

    while len(centers) < k:
        distances = np.min(np.sum((data_points - centers.reshape((len(centers), 1, -1))) ** 2, axis=2), axis=0)
        proportion = distances / np.sum(distances)
        centers = np.concatenate((centers, [data_points[np.random.choice(len(data_points), p=proportion)]]))

    return centers.reshape((k, 1, -1))


def k_means_clustering(data_points, k, initializer='random'):
    data_points = np.array(data_points)
    if initializer == 'random':
        while True:
            centers = data_points[np.random.choice(len(data_points), k)]
            if len(set(map(tuple, centers))) == k:
                centers = centers.reshape(k, 1, -1)
                break
    elif initializer == "k++":
        centers = k_means_initializer(data_points, k)
    else:
        centers = data_points[:k].reshape(k, 1, -1)
    while True:
        distances = np.sum((data_points - centers) ** 2, axis=2)
        idx = np.argmin(distances, axis=0)
        new_centers = []
        for i in range(k):
            cluster_points = data_points[np.where(idx == i)]
            new_centers.append(np.average(cluster_points, axis=0))

        dist = np.sum(centers.reshape(k, -1) - np.array(new_centers)) ** 2
        if dist == 0:
            break
        else:
            centers = np.array(new_centers).reshape((k, 1, -1))

    return np.round(centers.reshape(k, -1), 3)


def hidden_matrix_gravity(data_points, clusters):
    hidden_matrix = 1 / np.sum((data_points - clusters) ** 2, axis=2)
    hidden_matrix = hidden_matrix / hidden_matrix.sum(axis=0)
    return hidden_matrix


def hidden_matrix_partition(data_points, clusters, beta):
    hidden_matrix = np.exp(-(np.sum((data_points - clusters) ** 2, axis=2) ** 0.5 * beta))
    hidden_matrix = hidden_matrix / hidden_matrix.sum(axis=0)

    return hidden_matrix


def soft_k_means(data_points, k, initializer='random', beta=None, num_iter=100):
    data_points = np.array(data_points)
    low, high = np.min(data_points), np.max(data_points)
    if initializer == 'random':
        clusters = low + np.random.random((k, 2)) * (high - low)
    elif initializer == 'k++':
        clusters = k_means_initializer(data_points, k)
    else:
        clusters = data_points[:k]

    clusters = clusters.reshape(k, 1, -1)
    i = 0
    while i < num_iter:
        hidden_matrix = hidden_matrix_partition(data_points, clusters, beta).reshape(k, -1, 1)
        clusters = ((np.expand_dims(data_points, 0) * hidden_matrix)
                    .sum(axis=1) / hidden_matrix.sum(axis=1)).reshape(k, 1, -1)
        i += 1

    return clusters


def diff(s_snips, t_snips):
    num = len(t_snips[0])
    t_count = 0
    s_count = 0

    for snip_t in t_snips:
        for i in range(num - 1):
            for j in range(i + 1, num):
                if snip_t[i] != snip_t[j]:
                    t_count += 1
                    for snip_s in s_snips:
                        if snip_s[i] != snip_s[j]:
                            s_count += 1
                            break

    return s_count / t_count


def randomized_haplotype_search(s_snips, t_snips, k):
    s_snips = np.array(s_snips)
    t_snips = np.array(t_snips)
    snip_idx = np.random.choice(len(s_snips), k)
    best_snips = s_snips[snip_idx]
    best_score = diff(best_snips, t_snips)

    while True:
        current_snips = best_snips[::]
        for i in range(k):
            new_idx = snip_idx[::]
            new_idx = np.delete(new_idx, i)
            for j in range(len(s_snips)):
                if j not in snip_idx:
                    temp_idx = np.concatenate((new_idx, [j]))
                    score = diff(s_snips[temp_idx], t_snips)
                    if score > best_score:
                        best_score = score
                        best_snips = s_snips[temp_idx]

        if np.array_equal(current_snips, best_snips):
            break

    return best_snips


def verify_compatibility(snip_matrix: np.ndarray):
    # returns the lex sorted transposed snip matrix
    snip_matrix = np.transpose(snip_matrix)
    lex = ["".join(list(map(str, i))) for i in snip_matrix]
    sort_idx = np.argsort(lex)[::-1]
    snip_matrix = snip_matrix[sort_idx]
    for i in range(len(sort_idx) - 1):
        for j in range(i + 1, len(sort_idx)):
            dif_mat = snip_matrix[i] - snip_matrix[j]
            if np.sum(dif_mat) < 0:
                if np.sum(snip_matrix[j]) != np.sum(dif_mat == -1):
                    return None

    return snip_matrix


def perfect_phylogeny(snip_matrix):
    snip_matrix = np.array(snip_matrix)
    node_order = verify_compatibility(snip_matrix)
    clusters = [set([i for i in range(len(snip_matrix))])]
    graph = {tuple(clusters[0]): 0}

    for snip in node_order:
        ancestors = np.where(snip == 1)[0]
        for cluster in clusters:
            if ancestors[0] in cluster:
                new_cluster = set(tuple(ancestors))
                old_cluster = cluster.difference(new_cluster)
                clusters.remove(cluster)
                clusters += [new_cluster, old_cluster]
                cluster = tuple(cluster)
                if len(old_cluster) > 0:
                    new_node = tuple(
                        np.logical_and(snip_matrix[list(old_cluster)[0]],
                                       snip_matrix[list(new_cluster)[0]]).astype(int))
                    graph[tuple(old_cluster)] = new_node
                    graph[tuple(new_cluster)] = new_node
                    graph[new_node] = graph[tuple(cluster)]
                    graph.pop(cluster)

                break
    tree = defaultdict(list)
    for key, value in graph.items():
        if len(key) != snip_matrix.shape[1]:
            for k in key:
                tree[value].append(tuple(snip_matrix[k]))
        else:
            tree[value].append(key)

    return tree


def augment_perfect_phylogeny(graph, snip_vector):
    # Should come up with a better algorithm
    parent = graph[0][0]
    children = graph[parent]
    char_count = len(snip_vector)
    new_parent = None
    while True:
        for child in children:
            if child != parent:
                for i in range(char_count):
                    if child[i] == 1 and snip_vector[i] != 1:
                        break
                else:
                    new_parent = child
                    break

        if new_parent in graph and new_parent != parent:
            children = graph[new_parent]
            parent = new_parent
        elif new_parent != parent:
            new_node = tuple(np.logical_and(new_parent, snip_vector).astype(int))
            print('n', new_node)
            graph[parent].remove(new_parent)
            graph[parent].append(new_node)
            graph[new_node] += [tuple(snip_vector), new_parent]
            print('g', graph)
            break
        else:
            print(parent, new_parent)
            snip_vector = tuple(snip_vector)
            for node in children:
                if snip_vector == node:
                    graph[parent].remove(snip_vector)
                    graph[parent] += [snip_vector, snip_vector]
                    print(graph)
                    break
            else:
                new_node = np.logical_and(snip_vector, children[0]).astype(int)
                print('n', new_node)

            break
