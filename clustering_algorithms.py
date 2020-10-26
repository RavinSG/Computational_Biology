import numpy as np


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


def diff(snip_1, snip_2):
    num = len(snip_2[0])
    t_count = 0
    s_count = 0

    for snip_t in snip_2:
        for i in range(num - 1):
            for j in range(i + 1, num):
                if snip_t[i] != snip_t[j]:
                    t_count += 1
                    for snip_s in snip_1:
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
    snip_matrix = np.transpose(snip_matrix)
    lex = ["".join(list(map(str, i))) for i in snip_matrix]
    sort_idx = np.argsort(lex)

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
    leaves = [x for x in range(len(snip_matrix))]

    clusters = [set(leaves)]
    leaves = np.array(leaves)
    graph = {tuple(clusters[0]): 0}
    num_nodes = 1

    if node_order is not None:
        for i in node_order:
            cluster = leaves[np.where(i == 1)]
            for j in clusters:
                if cluster[0] in j:
                    parent_node = graph[tuple(j)]
                    cluster = set(cluster)
                    old_cluster = j
                    clusters.remove(old_cluster)
                    second_cluster = old_cluster.difference(cluster)

                    if len(second_cluster) > 0:
                        clusters += [cluster, second_cluster]
                        graph[tuple(second_cluster)] = num_nodes
                    else:
                        clusters += [cluster]

                    graph[num_nodes] = parent_node
                    graph.pop(tuple(j))
                    graph[tuple(cluster)] = num_nodes
                    num_nodes += 1

                    break

        return graph
    else:
        return None
