import numpy as np
from collections import defaultdict


def farthest_first_travel(data_points: list, k: int) -> np.ndarray:
    """
    Selects an arbitrary point in data as the first center and iteratively adds a new center as the point in data that
    is farthest from the centers chosen so far.

    :param data_points: A list of data points
    :param k: Number of centers
    :return: A list of centers
    """
    data_points = np.array(data_points)
    centers = np.array([data_points[0]])

    while len(centers) < k:
        distances = np.sum((data_points - centers.reshape((len(centers), 1, -1))) ** 2, axis=2) ** 0.5
        next_cluster = np.argmax(np.min(distances, axis=0))
        centers = np.concatenate((centers, [data_points[next_cluster]]))

    return centers


def max_distance(data_points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Finds the maximum distance to the closest center from all nodes
    """
    return np.max(
        np.min(np.sum((data_points - np.array(centers).reshape((len(centers), 1, -1))) ** 2, axis=2), axis=0) ** 0.5)


def calculate_distortion(data_points: np.ndarray, centers: np.ndarray) -> float:
    """
    Calculates the mean squared distance from each data point to its nearest center.
    """
    data_points = np.array(data_points)
    centers = np.array(centers).reshape((len(centers), 1, -1))

    distances = np.sum((data_points - centers.reshape((len(centers), 1, -1))) ** 2, axis=2)
    return np.min(distances, axis=0).sum() / len(data_points)


def k_means_initializer(data_points: np.ndarray, k: int) -> np.ndarray:
    """
    Chooses each point at random in such a way that distant points are more likely to be chosen than nearby points. The
    probability of selecting a center from data points is proportional to the squared distance of data points from the
    centers already chosen.

    :param data_points: A list of data points
    :param k: Number of centers
    :return: List of centers
    """
    centers = np.array([data_points[np.random.choice(len(data_points))]])

    while len(centers) < k:
        distances = np.min(np.sum((data_points - centers.reshape((len(centers), 1, -1))) ** 2, axis=2), axis=0)
        proportion = distances / np.sum(distances)
        centers = np.concatenate((centers, [data_points[np.random.choice(len(data_points), p=proportion)]]))

    return centers.reshape((k, 1, -1))


def k_means_clustering(data_points: list, k: int, initializer='random') -> np.ndarray:
    """
    Clusters the dataset using k-means clustering into k clusters.

    :param data_points: A list of data points
    :param k: Number of clusters
    :param initializer: Method to sample the initial centers
    :return: Centers of the clusters
    """
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


def hidden_matrix_gravity(data_points: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """
    Returns a len(clusters) x len(data) responsibility matrix for which [i, j] is the pull of center i on data point j.
    This pull is computed according to the Newtonian inverse-square law of gravitation
    """
    hidden_matrix = 1 / np.sum((data_points - clusters) ** 2, axis=2)
    hidden_matrix = hidden_matrix / hidden_matrix.sum(axis=0)
    return hidden_matrix


def hidden_matrix_partition(data_points: np.ndarray, clusters: np.ndarray, beta: float):
    """
    Returns a len(clusters) x len(data) responsibility matrix for which [i, j] is the partition value of center i on the
    data point j. The partition value is calculated using e to the power of -(distance to cluster i * stiffness
    parameter) divide by the sum across all clusters.

    The stiffness parameter is denoted by beta.
    """
    hidden_matrix = np.exp(-(np.sum((data_points - clusters) ** 2, axis=2) ** 0.5 * beta))
    hidden_matrix = hidden_matrix / hidden_matrix.sum(axis=0)

    return hidden_matrix


def soft_k_means(data_points: list, k: int, initializer='random', beta: float = None, num_iter=100) -> np.ndarray:
    """
    Initialize the algorithm with random centers, then assign each data point a responsibility for each cluster, where
    higher responsibilities correspond to stronger cluster membership. Then using the responsibilities of each data
    point create new clusters. Repeat process for num_iter iterations.

    :param data_points: A list of data points
    :param k: Number of clusters
    :param initializer: Method to sample the initial centers
    :param beta: Stiffness parameter
    :param num_iter: Number of iterations
    :return: Centers of the soft clusters
    """
    data_points = np.array(data_points)
    low, high = np.min(data_points), np.max(data_points)
    if initializer == 'random':
        clusters = low + np.random.random((k, 2)) * (high - low)
    elif initializer == 'k++':
        clusters = k_means_initializer(data_points, k)
    else:
        clusters = data_points[:k]

    clusters = clusters.reshape((k, 1, -1))
    i = 0
    while i < num_iter:
        hidden_matrix = hidden_matrix_partition(data_points, clusters, beta).reshape(k, -1, 1)
        clusters = ((np.expand_dims(data_points, 0) * hidden_matrix)
                    .sum(axis=1) / hidden_matrix.sum(axis=1)).reshape(k, 1, -1)
        i += 1

    return clusters


def diff(s_snips: np.ndarray, t_snips: np.ndarray) -> float:
    """
    Given two sets of snips S' and T, returns how well the collection of snips S’ explains a set T.
    Diff between two snips i and j is defined as follows.

    diff (s, t) = # of pairs of individuals (i, j) such that (s_i != s_j) and (t_i != t_j) /
                                # of pairs of (i, j) such that(t_i != t_j)

    The diff between the snip set S and a single snip t is defines as there is some SNP in S’ that explains this
    difference in t. This is then summed across all snips in T to calculate diff(S', T)
    """
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


def randomized_haplotype_search(s_snips: np.ndarray, t_snips: np.ndarray, k: int) -> np.ndarray:
    """
    Start with a random collection of k SNPs in S. At each step of the algorithm, every possible replacement of one SNP
    in the current collection with some SNP not in the collection and update S’ to be the set that maximizes Diff(S’, T)
    among all those considered. Repeat process till the score stops increasing.

    :param s_snips: Set of snips S
    :param t_snips: Set of snips T
    :param k: Number of snips in the collection
    :return: The list of snips in set S that explains T the most
    """
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


def verify_compatibility(snip_matrix: np.ndarray) -> np.ndarray:
    """
    Given a snip matrix checks whether all the columns in the matrix are compatible with each other. If two columns are
    compatible either one should be a subset of another, or both the columns should not share rows with 1 or the two
    columns should be the same.

    :param snip_matrix: A snip matrix where each column represents a character
    :return: The lex sorted transposed snip matrix
    """
    snip_matrix = np.transpose(snip_matrix)
    lex = ["".join(list(map(str, i))) for i in snip_matrix]
    sort_idx = np.argsort(lex)[::-1]
    snip_matrix = snip_matrix[sort_idx]
    for i in range(len(sort_idx) - 1):
        for j in range(i + 1, len(sort_idx)):
            dif_mat = snip_matrix[i] - snip_matrix[j]
            if np.sum(dif_mat) < 0:
                if np.sum(snip_matrix[j]) != np.sum(dif_mat == -1):
                    return np.ndarray([])

    return snip_matrix


def perfect_phylogeny(snip_matrix: np.ndarray) -> dict:
    """
    The columns of the snip matrix are treated as binary vectors, then the columns are sorting into descending
    lexicographic order from left to right. Then two children of the root are created and assign all members having
    the column as 1 to one child and all remaining members to the other node.

    Then this process is iterated, moving left to right within the columns of the snip matrix. When considering the
    i-th column, the algorithm moves downward in the tree, at each step choosing a child v if column i is a subset of
    the individuals contained in the subtree with the root as v.

    :param snip_matrix: A snip matrix where each column represents a character
    :return: A phylogenetic tree
    """
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
    """
    When a phylogeny graph is given with a new snip vector, travers the graph and add the snip to the proper location.
    """
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
