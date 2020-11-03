from copy import deepcopy
from collections import defaultdict

from params import *
from dna_replication import hamming_distance


def list_to_dict(adjacency_list: list, weighted=True) -> tuple:
    """
    Converts an adjacency list of an graph into a dictionary with first node of the edge as the key and the second node
    as another dictionary where the node is the key and the edge weight is the value. If the edges aren't weighted the
    value is set to 0. If the edges represent a directed graph, a dictionary containing the directed edges will also be
    returned.

    :param adjacency_list: A list of edges
    :param weighted: True if edges are weighted
    :return: The node dictionaries of the directed and undirected graphs
    """
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


def calculate_leaf_distance(adjacency_list: list) -> np.ndarray:
    """
    Calculate the distance between all the nodes using the Floyd-Warshall algorithm.

    :param adjacency_list: List of edges
    :return: A matrix containing the distance between leaves
    """
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


def limb_length(distances: np.ndarray, leaf: int) -> float:
    """
    Given a distance matrix between leaves, finds the length of the limb connecting leaf to the parent tree.

    :param distances: A distance matrix
    :param leaf: The node limb length should be calculated
    :return: The limb length
    """
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
    """
    Finds two nodes (i, j) such that they are located in two different subtrees that is joined by the parent node of n.

    :param matrix: A distance matrix
    :param n: Node dividing the two subtrees
    :param limb_len: Limb length of node n
    :return: Two nodes in the graph
    """
    for i in range(n):
        for j in range(n):
            if matrix[i][j] == matrix[i][n] + matrix[n][j] - limb_len * 2:
                return i, j

    return -1


def get_path(node_1: int, node_2: int, graph: dict) -> list:
    """
    Finds the path between two nodes in a tree graph
    """
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


def additive_phylogeny(distance_matrix: np.ndarray, num_nodes: int) -> tuple:
    """
    Using the distance matrix, creates a phylogeny tree using a bottom up approach. All internal nodes will be labeled
    with integers starting from the number of leaves in the tree.

    :param distance_matrix: Distance matrix between leaves
    :param num_nodes: Number of leave nodes
    :return: The phylogeny tree
    """
    n = len(distance_matrix) - 1
    if n == 1:
        return {0: {1: distance_matrix[0][1]}, 1: {0: distance_matrix[1][0]}}, num_nodes
    else:
        limb_len = limb_length(distance_matrix, n)

        # Select the final node and trim the tree
        tree, num_nodes = additive_phylogeny(distance_matrix[:-1, :][:, :-1], num_nodes)

        i, j = find_path_nodes(distance_matrix, n, limb_len)
        path = get_path(i, j, tree)
        dist = 0
        break_dist = distance_matrix[i, n] - limb_len
        # Find the location to connect the leaf to the parent tree
        for p in range(len(path) - 1):
            cur_node = path[p]
            next_node = path[p + 1]
            edge_distance = tree[cur_node][next_node]
            dist += edge_distance
            # If no node is present at the location break the edge and insert a node
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
            # Else add it to the existing node
            elif dist == break_dist:
                tree[next_node][n] = limb_len
                tree[n] = {next_node: limb_len}
                break

        return tree, num_nodes


def upgma(distance_matrix: np.ndarray, num_nodes: int) -> dict:
    """
    Using the distance matrix, creates a phylogeny tree using UPGMA (Unweighted Pair Group Method with Arithmetic Mean).
    All internal nodes will be labeled with integers starting from the number of leaves in the tree.

    :param distance_matrix: Distance matrix between leaves
    :param num_nodes: Number of leave nodes
    :return: The phylogeny tree
    """
    clusters = [(x,) for x in range(num_nodes)]
    graph = {(x,): [0] for x in range(num_nodes)}
    max_val = np.max(distance_matrix) + 1
    np.fill_diagonal(distance_matrix, max_val)

    # Run the algorithm till all nodes are clustered to one large cluster
    while len(clusters) > 1:
        i, j = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
        c_i = len(clusters[i])
        c_j = len(clusters[j])
        d_i_j = distance_matrix[i, j]

        # Calculate the distance between the new cluster and the existing clusters using the average
        col = (c_i * distance_matrix[:, i] + c_j * distance_matrix[:, j]) / (c_i + c_j)
        distance_matrix = np.delete(distance_matrix, [i, j], axis=0)
        distance_matrix = np.delete(distance_matrix, [i, j], axis=1)

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

        distance_matrix = np.hstack((distance_matrix, col.reshape(-1, 1)))
        distance_matrix = np.vstack((distance_matrix, np.append(col, max_val).reshape(1, -1)))
        clusters.append(new_cluster)
        num_nodes = num_nodes + 1

    return graph


def print_edge_distances(graph: dict, leaf_count: int):
    """
    Print all the edge distances of the graph in the format,
        node_1 -> node_2:distance
    It's assumed that leaves represent nucleotide sequences of length larger than 1
    """
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


def neighbour_join(distance_matrix: np.ndarray, nodes: list, num_leaves: int) -> dict:
    """
    Given an additive distance matrix, neighbour_join finds a pair of neighboring leaves and substitutes them by a
    single leaf, thus reducing the size of the tree. neighbour_join can thus recursively construct a tree fitting the
    additive matrix.

    Although finding a minimum element in a distance matrix D is not guaranteed to yield a pair of neighbors in the
    tree, the matrix is transformed into a neighbour-joining-matrix.

    :param distance_matrix: Distance matrix between leaves
    :param nodes: A list of nodes
    :param num_leaves: Number of leaves
    :return: The neighbour joined tree
    """
    n = len(distance_matrix)
    if n == 2:
        dist = np.array(distance_matrix)
        return {nodes[0]: {nodes[1]: dist}, nodes[1]: {nodes[0]: dist}}
    else:
        total_distance = distance_matrix.sum(axis=1)
        d_star = ((n - 2) * distance_matrix - total_distance) - total_distance.reshape(-1, 1)
        max_val = max(d_star.max() + 1, 0)
        np.fill_diagonal(d_star, max_val)
        i, j = np.unravel_index(np.argmin(d_star), d_star.shape)
        node_i = nodes[i]
        node_j = nodes[j]
        nodes.remove(node_i)
        nodes.remove(node_j)
        d_min = distance_matrix[i][j]
        delta = (total_distance[i] - total_distance[j]) / (n - 2)

        limb_i = (d_min + delta) / 2
        limb_j = (d_min - delta) / 2

        distance_k = np.delete((distance_matrix[:, j] + distance_matrix[:, i] - d_min) / 2, [i, j])
        distance_matrix = np.delete(distance_matrix, [i, j], axis=0)
        distance_matrix = np.delete(distance_matrix, [i, j], axis=1)

        distance_matrix = np.hstack((distance_matrix, distance_k.reshape(-1, 1)))
        distance_matrix = np.vstack((distance_matrix, np.append(distance_k, max_val).reshape(1, -1)))
        nodes.append(num_leaves + 1)
        tree = neighbour_join(distance_matrix, nodes, num_leaves + 1)
        for x, y in zip([node_i, node_j], [limb_i, limb_j]):
            tree[num_leaves + 1][x] = y
            tree[x] = {num_leaves + 1: y}

        return tree


def small_parsimony(graph: dict, node, char_num: int, s_k_values: dict) -> tuple:
    """
    Calculates the s_k values of nodes of a tree. Where s_k value is defined as,
            s_k(v) = min all symbols i{s_i(DAUGHTER(v)) + d_i, k} + min all symbols j{s_j(SON(v)) + d_j, k}

    The graph should be in the format of: {
                                            parent_1: {child_1,child_2},
                                            parent_2: {child_3,child_4},
                                            ...
                                        }

    :param graph: The graph in a dictionary format
    :param node: Node parsimony score should
    :param char_num: Position of the sequence
    :param s_k_values: s_k value of the node
    :return: The s_k values of all nodes
    """
    if not node.isdigit():
        s_k = np.ones(4) * np.inf
        s_k[STR_TO_NUM[node[char_num]]] = 0
        s_k_values[node] = s_k
        return s_k, s_k_values

    else:
        s_k = 0
        for child in graph[node]:
            child_sk, s_k_values = small_parsimony(graph, child, char_num, s_k_values)
            alpha = np.ones((4, 4))
            np.fill_diagonal(alpha, 0)
            s_k += (child_sk + alpha).min(axis=1)
        s_k_values[node] = s_k
        return s_k, s_k_values


def backtrack_parsimony(graph: dict, s_k_values: dict, node) -> dict:
    """
    Converts the values of the s_k_values dictionary to characters that minimizes the parsimony score of the tree.

    :param graph: The graph in a dictionary format
    :param s_k_values: s_k value of the node
    :param node: Node a character should be assigned
    :return: Character assigned tree
    """
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


def create_character_sequence(graph: dict, root, leaf_len: int) -> tuple:
    """
    Given a rooted binary tree with each leaf labeled by a string of length m, returns a labeling of all other nodes of
    the tree by strings of length m that minimizes the tree’s parsimony score.

    The parsimony score of a tree is the sum of the lengths of its edges, where the length of an edge corresponds to the
    hamming distance between the nodes.

    :param graph: The graph in a dictionary format
    :param root: Root node of the graph
    :param leaf_len: Length of the sequence in the leaf node
    :return: Rooted binary tree with the lowest parsimony score
    """
    trees = []
    for i in range(leaf_len):
        trees.append(small_parsimony(graph, root, i, {})[1])

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


def print_all_edges(graph: dict, string_mapping: dict, skip_root=None):
    """
    Given a tree and a node to string mapping, prints all the nodes in the format of,
                    node_1->node_2:hamming_distance(node_1,node_2)
    """
    nodes = [list(graph)[0]]
    # For an un-rooted binary tree, skips printing the pseudo root node
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


def insert_first_root(adjacency_list: list) -> tuple:
    """
    Given an edge lest of an un-rooted binary tree, arbitrary pick an edge and root the tree breaking the edge.
    """
    new_root = str(len(adjacency_list) // 2 + 1)
    old_edge = adjacency_list[0]
    node_1, node_2 = old_edge.split('->')
    adjacency_list.remove(old_edge)
    adjacency_list.remove('->'.join([node_2, node_1]))
    adjacency_list += ['->'.join([new_root, node]) for node in [node_1, node_2]]
    adjacency_list += ['->'.join([node, new_root]) for node in [node_1, node_2]]

    return adjacency_list, new_root


def directed_binary_tree(adjacency_list, root=None) -> tuple:
    """
    Converts an undirected rooted binary to a directed tree and returns the directed graph with the root node.
    """
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


def move_sub_tree(parents, root, start_node, parent_node, sibling, direction, leaf_len):
    """
    Moves the root from parent of the node, to the edge between the node and it's children and create a tree for each
    edge. Recursively traverse all the nodes switching the root to the nodes children. This will generate all possible
    rooted trees with only O(num_nodes) time complexity.

             root                          new_root                                new_root
              |                            /    \                                  /    \
              |                           /      \                                /      \
        parent_node        --->      child_1   parent_node        and      parent_node   child_2
           /    \                                  \                           /
          /      \                                  \                         /
      child_1  child_2                            child_2                 child_1

    :param parents: Parent dictionary of the graph
    :param root: Root of the previous tree
    :param start_node: Node on the other side of the end the root should be passed onto
    :param parent_node: Child of the current root node
    :param sibling: Other child of the root node
    :param direction: The direction the root is traversing
    :param leaf_len: Length of the sequence in the leaf node
    :return: The tree with the lowest parsimony score in the rooted sub-tree
    """
    if start_node in parents:
        l_child, r_child = parents[start_node]
        if direction == 'left':
            left_parents = deepcopy(parents)
            left_parents[parent_node].pop(start_node)
            left_parents[root][start_node] = 0
            left_parents[root][parent_node] = 0
            left_parents[parent_node] = {sibling: 0, list(left_parents[parent_node])[0]: 0}

            tree_m = deepcopy(left_parents)
            score, str_mapping = create_character_sequence(left_parents, root, leaf_len)
            left_parents.pop(root)
            s_1, tree_l = move_sub_tree(left_parents, root, l_child, start_node, parent_node, 'left', leaf_len)
            s_2, tree_r = move_sub_tree(left_parents, root, r_child, start_node, parent_node, 'right', leaf_len)

            scores = [score, s_1, s_2]
            trees = [[tree_m, str_mapping], tree_l, tree_r]
            lowest_idx = int(np.argmin(scores))
            return scores[lowest_idx], trees[lowest_idx]
        else:
            right_parents = deepcopy(parents)
            right_parents[parent_node].pop(start_node)
            right_parents[root][parent_node] = 0
            right_parents[root][start_node] = 0
            right_parents[parent_node] = {list(right_parents[parent_node])[0]: 0, sibling: 0}

            tree_m = deepcopy(right_parents)
            score, str_mapping = create_character_sequence(right_parents, root, leaf_len)
            right_parents.pop(root)
            s_1, tree_r = move_sub_tree(right_parents, root, r_child, start_node, parent_node, 'right', leaf_len)
            s_2, tree_l = move_sub_tree(right_parents, root, l_child, start_node, parent_node, 'left', leaf_len)

            scores = [score, s_1, s_2]
            trees = [[tree_m, str_mapping], tree_r, tree_l]
            lowest_idx = int(np.argmin(scores))
            return scores[lowest_idx], trees[lowest_idx]
    else:
        # if the child is a leaf node stop the traversal
        node_parents = deepcopy(parents)
        node_parents[parent_node].pop(start_node)
        node_parents[root][start_node] = 0
        node_parents[root][parent_node] = 0
        node_parents[parent_node][sibling] = 0
        score, str_mapping = create_character_sequence(node_parents, root, leaf_len)
        return score, [node_parents, str_mapping]


def move_root(parent_graph: dict, old_root, leaf_len: int) -> tuple:
    """
    Given a directed rooted binary tree with a pseudo node, where the position of the root in the tree is unknown,
    assign the root to each edge, apply small-parsimony to the resulting tree.

    Out of all the created trees find the labeling of all other nodes that minimizes the parsimony score and return the
    un-rooted tree relevant for the lowest score.

    :param parent_graph: Parent dictionary of the graph
    :param old_root: Root of the graph
    :param leaf_len: Length of the sequence in the leaf node
    :return: The minimum parsimony score and the relevant un-rooted tree
    """
    best_tree = deepcopy(parent_graph)
    score, str_mapping = create_character_sequence(parent_graph, old_root, leaf_len)
    best_tree = [best_tree, str_mapping]

    left_node = list(parent_graph[old_root])[0]
    right_node = list(parent_graph[old_root])[1]

    parent_graph.pop(old_root)
    left_parents = deepcopy(parent_graph)
    right_parents = deepcopy(parent_graph)
    dirs = ['left', 'right']

    # Since there are four directions the root can travel, find the best score of each direction
    for child, direction in zip(parent_graph[left_node], dirs):
        result = move_sub_tree(left_parents, old_root, child, left_node, right_node, direction, leaf_len)
        if result[0] < score:
            score = result[0]
            best_tree = result[1]

    for child, direction in zip(parent_graph[right_node], dirs):
        result = move_sub_tree(right_parents, old_root, child, right_node, left_node, direction, leaf_len)
        if result[0] < score:
            score = result[0]
            best_tree = result[1]

    return score, best_tree


def nearest_neighbours(edge: set, adjacency_list: list) -> list:
    """
    The neighbours of an edge is defined as follows. An internal edge is an between any two nodes which are not leaves.

    For a given internal edge (a, b), denote the remaining nodes adjacent to a as w and x; and denote the remaining
    nodes adjacent to b as y and z.
    First neighbour is obtained by removing edges (a, x) and (b, y) and replacing them with (a, y) and (b, x).
    Second neighbour is obtained by  removing edges (a, x) and (b, z) and replacing them with (a, z) and (b, x)

    The adjacency list should be in the format of [node_1->node_2, node_2->node_3, node_2->node_4,...]

    :param edge: An internal edge of the graph
    :param adjacency_list: List of edges in the graph
    :return: The two neighbours respective to the edge
    """
    temp_list = adjacency_list.copy()
    node_1, node_2 = edge
    neigh_1 = []
    neigh_2 = []

    for i in temp_list:
        nodes = i.split('->')
        if nodes[0] == node_1:
            if nodes[1] != node_2:
                neigh_1.append(nodes[1])

        elif nodes[0] == node_2:
            if nodes[1] != node_1:
                neigh_2.append(nodes[1])

    # Since the graph is undirected, edges in both directions should be deleted
    for i in neigh_1:
        temp_list.remove(f"{i}->{node_1}")
        temp_list.remove(f"{node_1}->{i}")

    for i in neigh_2:
        temp_list.remove(f"{i}->{node_2}")
        temp_list.remove(f"{node_2}->{i}")

    neighbours = []
    for i in range(2):
        node_list = temp_list[::]
        node_list += [f'{node_1}->{neigh_1[0]}']
        node_list += [f'{neigh_1[0]}->{node_1}']

        node_list += [f'{node_1}->{neigh_2[i]}']
        node_list += [f'{neigh_2[i]}->{node_1}']

        node_list += [f'{node_2}->{neigh_1[1]}']
        node_list += [f'{neigh_1[1]}->{node_2}']

        node_list += [f'{node_2}->{neigh_2[1 - i]}']
        node_list += [f'{neigh_2[1 - i]}->{node_2}']

        neighbours.append(node_list)

    return neighbours


def dict_to_list(graph: dict, un_root=None) -> list:
    """
    Converts a graph from dictionary format to an adjacency list.
    """
    adjacency_list = []
    if un_root is not None:
        children = list(graph.pop(un_root))
        graph[children[0]][children[1]] = 0

    for node in graph:
        for child in graph[node]:
            adjacency_list.append(f"{child}->{node}")
            adjacency_list.append(f"{node}->{child}")

    return adjacency_list


def nearest_neighbour_interchange(adjacency_list: list, leaf_len: int, print_intermediate=False) -> tuple:
    """
    Tries to find a solution for the large parsimony problem using the nearest neighbour interchange as a heuristic.
    Initially assigns input strings to arbitrary leaves of the tree, assigns strings to the internal nodes of the tree
    by solving the Small Parsimony Problem in an un-rooted tree, and then moves to a nearest neighbor that provides the
    best improvement in the parsimony score.

    At each iteration, the algorithm explores all internal edges of a tree and generates all nearest neighbor
    interchanges for each internal edge. For each of these nearest neighbors, the algorithm solves the Small Parsimony
    Problem to reconstruct the labels of the internal nodes and computes the parsimony score.

    If a nearest neighbor with smaller parsimony score is found, then the algorithm. selects the one with smallest
    parsimony score and iterates again; otherwise, the algorithm terminates.

    :param adjacency_list: Edge list of the graph
    :param leaf_len: Length of the sequence in the leaf node
    :param print_intermediate: If true, prints the best after each iteration
    :return: A tuple containing the best parsimony score, the tree responsible for the best score, the character
    sequences of internal nodes and the adjacency list of the graph
    """
    best_score = np.inf
    best_graph = None
    best_mapping = None

    adjacency_list, root = insert_first_root(adjacency_list)
    parent_graph, _ = directed_binary_tree(adjacency_list, root)
    score, tree = move_root(parent_graph, root, leaf_len)
    tree_graph, str_mapping = tree
    best_adjacency_list = dict_to_list(tree_graph, root)

    while score < best_score:
        best_score = score
        best_mapping = str_mapping
        best_graph = tree_graph
        swapped_edges = []
        adjacency_list = best_adjacency_list.copy()
        for edge in adjacency_list:
            nodes = set(x for x in edge.split('->'))
            if nodes not in swapped_edges:
                swapped_edges.append(nodes)
                for i in nodes:
                    if not i.isdigit():
                        break
                else:
                    neighbours = nearest_neighbours(nodes, adjacency_list)
                    for neighbour in neighbours:
                        neighbour, root = insert_first_root(neighbour)
                        parent_graph, _ = directed_binary_tree(neighbour, root)
                        new_score, new_tree = move_root(parent_graph, root, leaf_len)
                        new_graph, new_mapping = new_tree
                        if new_score < score:
                            score = new_score
                            tree_graph = new_graph
                            str_mapping = new_mapping
                            best_adjacency_list = dict_to_list(tree_graph, root)

        if print_intermediate:
            print(int(score))
            for adj_edge in best_adjacency_list:
                ends = adj_edge.split('->')
                print(
                    f"{str_mapping[ends[0]]}->{str_mapping[ends[1]]}:"
                    f"{hamming_distance(str_mapping[ends[0]], str_mapping[ends[1]])}")
            print('\n')

    return best_score, best_graph, best_mapping, best_adjacency_list
