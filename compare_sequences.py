from collections import OrderedDict

from params import *


def min_num_coins(money: int, coins: list) -> dict:
    """
    Given a amount of money and a list of coins, returns the minimum number of coins needed to pay the amount. This uses
    constant amount of space.

    :param money: The value to be created
    :param coins: Possible coin values
    :return: A dictionary containing the number of coins needed for each value between money and money-max(coins)
    """
    max_coin = max(coins)
    cache = OrderedDict({0: 0})

    for m in range(1, money + 1):
        try:
            cache[m] = min([cache[m - c] for c in coins if c <= m]) + 1
            while len(cache) > max_coin:
                cache.popitem(False)

        except ValueError:
            cache[m] = np.inf

    return cache


def manhattan_tourist_problem(n: int, m: int, down_matrix: np.ndarray, right_matrix: np.ndarray) -> float:
    """
    Given a manhattan grid of size (n x m) and the values of each path down and right, returns the maximum achievable
    score by starting from the (0,0) and traveling to (n,m) by only travelling down or right at each intersection.

    :param n: Number of rows
    :param m: Number of columns
    :param down_matrix: Values of the down action at each intersection
    :param right_matrix: Values of the right action at each intersection
    :return: The maximum achievable score
    """
    distances = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        distances[i, 0] = distances[i - 1, 0] + down_matrix[i - 1, 0]

    for i in range(1, m + 1):
        distances[0, i] = distances[0, i - 1] + right_matrix[0, i - 1]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            distances[i, j] = max((distances[i - 1, j] + down_matrix[i - 1, j]),
                                  (distances[i, j - 1] + right_matrix[i, j - 1]))

    return distances[-1, -1]


def string_backtrack(string_1: str, string_2: str, score_matrix: dict = None, indel_penalty=5, local_align=False,
                     fitting=False) -> tuple:
    """
    Given two string and a score matrix, returns the backtrack and score matrices of the two strings. If a score matrix
    is not given each match will have a value of 1, indel and mismatch penalties will be 0. The backtrack matrix,
    score matrix and the highest score will be returned.

    If local alignment is true the score will be calculated for to maximize local alignment instead of a global
    alignment. A local alignment is defined as an alignment between the two strings starting and ending from any
    position.

    If fitting is true, both whole strings will be aligned, but unlike global alignment the alignment doesn't need to
    cover the whole string length. The shorter string should be the second string.

                               Global              Local              Fitting
    GTAGGCTTAAGGTTA  :     GTAGGCTTAAGGTTA     GTAGGCTTAAGGTTA     GTAGGCTTAAGGTTA
    TAGATA           :     -TAG----A---T-A      TAG                 TAGA--TA

    :param string_1: String to be aligned
    :param string_2: String to be aligned
    :param score_matrix: This will be used to score the alignment of each letters
    :param indel_penalty: Penalty for inserting or deleting letters
    :param local_align: If true local alignment will be used
    :param fitting: If true fitting alignment will be used
    :return: A tuple containing the backtrack matrix, score matrix and the maximum score
    """
    l_1 = len(string_1) + 1
    l_2 = len(string_2) + 1

    max_values = np.zeros((l_1, l_2))
    backtrack = np.zeros((l_1, l_2))
    base_value = -np.inf
    if local_align:
        # Chane the default score of every node to 0 to indicate a direct edge from the starting node.
        base_value = 0
    if score_matrix is None:
        for i in range(1, l_1):
            for j in range(1, l_2):
                match = 0
                if string_1[i - 1] == string_2[j - 1]:
                    match = 1

                top = max_values[i - 1, j]
                left = max_values[i, j - 1]
                diag = max_values[i - 1, j - 1]

                max_values[i, j] = max(top, left, diag + match, base_value)

                if max_values[i, j] == top:
                    backtrack[i, j] = 0
                elif max_values[i, j] == left:
                    backtrack[i, j] = 1
                else:
                    backtrack[i, j] = 2
    else:
        if not fitting:
            # If fitting the starting node of every column is set to 0 to indicate that the second string can start
            # from any location in the first string.
            for i in range(1, l_1):
                max_values[i, 0] = max_values[i - 1, 0] - indel_penalty

        for i in range(1, l_2):
            max_values[0, i] = max_values[0, i - 1] - indel_penalty

        for i in range(1, l_1):
            for j in range(1, l_2):

                top = max_values[i - 1, j] - indel_penalty
                left = max_values[i, j - 1] - indel_penalty
                diag = max_values[i - 1, j - 1] + score_matrix[string_1[i - 1]][string_2[j - 1]]

                max_values[i, j] = max(top, left, diag, base_value)

                if max_values[i, j] == top:
                    backtrack[i, j] = 0
                elif max_values[i, j] == left:
                    backtrack[i, j] = 1
                else:
                    backtrack[i, j] = 2

    score = max_values[-1, -1]
    return backtrack, max_values, score


def find_longest_common_sequence(string_1: str, string_2: str, score_matrix: dict = None, indel_penalty=0) -> tuple:
    """
    Given two strings and a score matrix, returns the alignment of the two strings that maximizes the score respective
    to the given score matrix.

    :param string_1: String to be aligned
    :param string_2: String to be aligned
    :param score_matrix: This will be used to score the alignment of each letters
    :param indel_penalty: Penalty for inserting or deleting letters
    :return: A tuple containing the alignment of the two strings and the score
    """
    backtrack, _, score = string_backtrack(string_1, string_2, score_matrix, indel_penalty=indel_penalty)
    i = len(string_1)
    j = len(string_2)
    align_1 = ""
    align_2 = ""
    while i > 0:
        value = backtrack[i, j]
        if value == 0:
            align_1 = string_1[i - 1] + align_1
            align_2 = "-" + align_2
            i = i - 1
        elif value == 1:
            align_1 = "-" + align_1
            align_2 = string_2[j - 1] + align_2
            j = j - 1
        else:
            align_1 = string_1[i - 1] + align_1
            align_2 = string_2[j - 1] + align_2
            i = i - 1
            j = j - 1

    # If the alignment of the stings end before reaching the end, add indels to cover the gap
    while j > 0:
        align_1 = "-" + align_1
        align_2 = string_2[j - 1] + align_2
        j -= 1

    return align_1, align_2, score


def find_local_alignment(string_1: str, string_2: str, score_matrix: dict, indel_penalty=5) -> tuple:
    """
    Given two strings and a score matrix, returns the local alignment of the two strings that maximizes the score
    respective to the given score matrix.

    :param string_1: String to be aligned
    :param string_2: String to be aligned
    :param score_matrix: This will be used to score the alignment of each letters
    :param indel_penalty: Penalty for inserting or deleting letters
    :return: A tuple containing the alignment of the two strings
    """
    backtrack, max_values, _ = string_backtrack(string_1, string_2, score_matrix, local_align=True,
                                                indel_penalty=indel_penalty)
    align_1 = ""
    align_2 = ""

    # Find the node with the maximum score and backtrack till a node of value 0 is reached
    end_node = np.unravel_index(np.argmax(max_values), max_values.shape)
    i, j = end_node
    while i > 0:
        value = backtrack[i, j]
        if value == 0:
            align_1 = string_1[i - 1] + align_1
            align_2 = "-" + align_2
            i = i - 1
        elif value == 1:
            align_1 = "-" + align_1
            align_2 = string_2[j - 1] + align_2
            j = j - 1
        else:
            if max_values[i, j] == 0:
                score = max_values[end_node[0], end_node[1]]
                return align_1, align_2, score
            else:
                align_1 = string_1[i - 1] + align_1
                align_2 = string_2[j - 1] + align_2
                i = i - 1
                j = j - 1

    return align_1, align_2


def find_longest_path(node: int, parents: dict, path_lengths: dict, backtrack: dict) -> tuple:
    """
    Given a node and a graph in the format of {node1: parent1, node2: parent2, node3:parent1,..} format, along with a
    path length dictionary that contains the maximum distance to each node from the source node, and a backtrack
    dictionary, returns a tuple containing the max distance, path length dictionary and the backtrack dictionary.

    :param node: The node distance should be calculated to from the source
    :param parents: A graph in dictionary format
    :param path_lengths: A dictionary that contains the distance from the source to each node, {source:0}
    :param backtrack: Dictionary containing backtrack pointers
    :return: A tuple containing the max distance, path length dictionary and the backtrack dictionary
    """
    if node in path_lengths:
        return path_lengths[node], path_lengths, backtrack
    else:
        length = -np.inf
        parent_node = None
        if node in parents:
            for parent in parents[node]:
                parent_len, _, _ = find_longest_path(parent[0], parents, path_lengths, backtrack)
                parent_len += parent[1]
                if parent_len > length:
                    length = parent_len
                    parent_node = parent

        path_lengths[node] = length
        if parent_node is not None:
            backtrack[node] = parent_node[0]

    return path_lengths[node], path_lengths, backtrack


def calculate_global_score(string_1: str, string_2: str, score_matrix: dict, indel_penalty: float) -> float:
    """
    Calculates the score of the global alignment of two strings respective to the score matrix and indel penalty.

    :param string_1: Aligned string
    :param string_2:  Aligned string
    :param score_matrix: This will be used to score the alignment of each letters
    :param indel_penalty: Penalty for inserting or deleting letters
    :return: The score of the alignment
    """
    score = 0
    for x, y in zip(string_1, string_2):
        if x == '-' or y == '-':
            score -= indel_penalty
        else:
            score += score_matrix[x][y]

    return score


def generate_score_matrix(string_1: str, string_2: str, sim_score=1, mismatch=1) -> dict:
    """
    Using the characters in the two strings as the alphabet, creates a score matrix that has similarity score for
    matching elements and the mismatch penalty for every other combination.

    :param string_1: String of characters
    :param string_2: String of characters
    :param sim_score: The value of the score matrix if the symbols match
    :param mismatch: The value of the score matrix if symbols don't match
    :return: The score matrix
    """
    alphabet = set(string_1).union(string_2)
    score_matrix = {}
    for i in alphabet:
        score_matrix[i] = {x: sim_score if x == i else -mismatch for x in alphabet}

    return score_matrix


def edit_distance(string_1: str, string_2: str) -> int:
    """
    Given two strings calculate the minimum number of edits that should be made in one string to make it equivalent to
    the other string.

    :param string_1: String of characters
    :param string_2: String of characters
    :return: The edit distance between the two stings
    """
    score_matrix = generate_score_matrix(string_1, string_2)
    align_1, align_2, _ = find_longest_common_sequence(string_1, string_2, score_matrix, indel_penalty=1)
    distance = 0

    for i in range(len(align_1)):
        if align_1[i] != align_2[i]:
            distance += 1
    return distance


def fitting_alignment(string_1: str, string_2: str, overlapping=False) -> tuple:
    """
    Given two strings , returns the fitting alignment of the two strings that maximizes the score of alignment
    respective to the score matrix generated from the two strings.

    If overlapping is true, the alignment will be an overlapping of the end of the first string with the beginning of
    the second string.
                      e.g:  ATGCATGCCGG
                                 T-CC-GAAAC

    :param string_1: String to be aligned
    :param string_2: String to be aligned
    :param overlapping: If true the ends will be overlapped
    :return:
    """
    align_1 = ""
    align_2 = ""
    score_matrix = generate_score_matrix(string_1, string_2)
    backtrack, max_values, _ = string_backtrack(string_1, string_2, score_matrix, local_align=False, indel_penalty=2,
                                                fitting=True)

    if overlapping:
        i = len(string_1)
        j = np.argmax(max_values[-1])

    else:
        i = np.argmax(max_values, axis=0)[-1]
        j = len(string_2)

    end_row = [i, j]

    while j > 0:
        value = backtrack[i, j]
        if value == 0:
            align_1 = string_1[i - 1] + align_1
            align_2 = "-" + align_2
            i = i - 1
        elif value == 1:
            align_1 = "-" + align_1
            align_2 = string_2[j - 1] + align_2
            j = j - 1
        else:
            align_1 = string_1[i - 1] + align_1
            align_2 = string_2[j - 1] + align_2
            i = i - 1
            j = j - 1

    score = max_values[end_row[0], end_row[1]]

    return align_1, align_2, score


def backtrack_layers(string_1, string_2, upper, middle, lower, local=False, end_node=None):
    """
    Given two strings with the three backtrack matrices relevant to each layer align the two strings accordingly. If
    local is true, local alignment will be used instead of global. The end node should be the final node of the local
    alignment.

    :param string_1: String to be aligned
    :param string_2: String to be aligned
    :param upper: Backtrack matrix for the nodes of the top upper layer
    :param middle: Backtrack matrix for the nodes of the top middle layer
    :param lower: Backtrack matrix for the nodes of the top lowe layer
    :param local: If true local alignment is used
    :param end_node: The final node of the local alignment
    :return: A tuple containing the aligned two strings
    """
    if local:
        i, j = end_node
    else:
        i = len(string_1)
        j = len(string_2)
    align_1 = ""
    align_2 = ""
    current_layer = middle
    # Layer ids: {upper:0, middle:1, lower:2}
    layer_id = 1

    while i > 0:
        value = current_layer[i, j]
        if layer_id == 1:
            if value == 0:
                current_layer = upper
                layer_id = 0

            elif value == 1:
                current_layer = lower
                layer_id = 2

            elif value == 2:
                align_1 = string_1[i - 1] + align_1
                align_2 = string_2[j - 1] + align_2
                i -= 1
                j -= 1
            else:
                # Value 3 represents an edge to the starting node
                align_1 = string_1[i - 1] + align_1
                align_2 = string_2[j - 1] + align_2
                return align_1, align_2

        elif layer_id == 0:
            align_1 = string_1[i - 1] + align_1
            align_2 = "-" + align_2
            i -= 1
            if value == 0:
                continue
            elif value == 1:
                current_layer = middle
                layer_id = 1
            else:
                return align_1, align_2

        else:
            align_1 = '-' + align_1
            align_2 = string_2[j - 1] + align_2
            j -= 1
            if value == 0:
                continue
            elif value == 1:
                current_layer = middle
                layer_id = 1
            else:
                return align_1, align_2

    return align_1, align_2


def affine_gap_penalty(string_1: str, string_2: str, score_matrix: dict, gap_penalty: float, ext_penalty: float,
                       local=False) -> tuple:
    """
    Finds the highest scoring global alignment between the two strings, as defined by the score matrix and by the gap
    opening and extension penalties.

    :param string_1: String to be aligned
    :param string_2: String to be aligned
    :param score_matrix: This will be used to score the alignment of each letters
    :param gap_penalty: The penalty for opening a gap
    :param ext_penalty: The penalty for extending a gap
    :param local: If true local alignment will be used
    :return: The rwo aligned string
    """
    l_1 = len(string_1) + 1
    l_2 = len(string_2) + 1
    base_val = -np.inf
    if local:
        base_val = 0

    upper_vertical = np.zeros((l_1, l_2))
    middle = np.zeros((l_1, l_2))
    lower_horizontal = np.zeros((l_1, l_2))

    backtrack_u = np.zeros((l_1, l_2))
    backtrack_m = np.zeros((l_1, l_2))
    backtrack_l = np.zeros((l_1, l_2))

    for i in range(1, l_1):
        upper_vertical[i, 0] = -(gap_penalty + (i - 1) * ext_penalty)
        middle[i, 0] = -(gap_penalty + (i - 1) * ext_penalty)
        lower_horizontal[i, 0] = -(gap_penalty + (i - 1) * ext_penalty)

    for i in range(1, l_2):
        upper_vertical[0, i] = -(gap_penalty + (i - 1) * ext_penalty)
        middle[0, i] = -(gap_penalty + (i - 1) * ext_penalty)
        lower_horizontal[0, i] = -(gap_penalty + (i - 1) * ext_penalty)

    for i in range(1, l_1):
        for j in range(1, l_2):
            upper_val = [upper_vertical[i - 1, j] - ext_penalty, middle[i - 1, j] - gap_penalty, base_val]
            lower_val = [lower_horizontal[i, j - 1] - ext_penalty, middle[i, j - 1] - gap_penalty, base_val]

            actions = np.argmax([upper_val, lower_val], axis=1)

            upper_vertical[i, j] = upper_val[actions[0]]
            lower_horizontal[i, j] = lower_val[actions[1]]

            middle_val = [upper_vertical[i, j], lower_horizontal[i, j],
                          middle[i - 1, j - 1] + score_matrix[string_1[i - 1]][string_2[j - 1]],
                          base_val + score_matrix[string_1[i - 1]][string_2[j - 1]]]

            # print(string_1[i - 1], string_2[j - 1], score_matrix[string_1[i - 1]][string_2[j - 1]])
            action_m = int(np.argmax(middle_val))
            # print(action_m, middle_val)
            middle[i, j] = middle_val[action_m]

            backtrack_u[i, j] = actions[0]
            backtrack_l[i, j] = actions[1]
            backtrack_m[i, j] = action_m

    score = middle[-1, -1]
    end_node = None
    if local:
        end_node = np.unravel_index(np.argmax(middle), middle.shape)
        score = np.max(middle)

    align_1, align_2 = backtrack_layers(string_1, string_2, backtrack_u, backtrack_m, backtrack_l, local=local,
                                        end_node=end_node)

    return align_1, align_2, score


def calculate_source_i(string_1: str, string_2: str, score_matrix: dict, indel_penalty: float) -> np.ndarray:
    """
    Uses linear additional memory, calculates the score of the nodes of the graph created by the two strings, and
    returns the score of the final two columns. The matrix is of shape of (len(string_1) x 2).

    :param string_1: String to be aligned
    :param string_2: String to be aligned
    :param score_matrix: This will be used to score the alignment of each letters
    :param indel_penalty: Penalty for inserting or deleting letters
    :return: A two column matrix containing the scores of the nodes
    """
    l_1 = len(string_1) + 1
    l_2 = len(string_2) + 1

    max_values = np.zeros((l_1, 2))

    for i in range(1, l_1):
        max_values[i, 0] = max_values[i - 1, 0] - indel_penalty

    for j in range(1, l_2):
        max_values[0, j % 2] = max_values[0, (j + 1) % 2] - indel_penalty
        for i in range(1, l_1):
            top = max_values[i - 1, j % 2] - indel_penalty
            left = max_values[i, (j - 1) % 2] - indel_penalty
            diag = max_values[i - 1, (j - 1) % 2] + score_matrix[string_1[i - 1]][string_2[j - 1]]

            max_values[i, j % 2] = max(top, left, diag)

    if l_2 % 2 == 1:
        max_values = max_values[:, [1, 0]]
    return max_values


def find_middle_edge(string_1: str, string_2: str, score_matrix: dict, indel_penalty: float) -> tuple:
    # global scores
    """
    A middle edge in the alignment graph of these strings where the edge lengths are defined by score matrix.

    :param string_1: String to be aligned
    :param string_2: String to be aligned
    :param score_matrix: This will be used to score the alignment of each letters
    :param indel_penalty: Penalty for inserting or deleting letters
    :return: The middle edge and the backtrack pointer for the edge
    """
    mid_point = np.floor(len(string_2) / 2).astype(int)

    a = calculate_source_i(string_1, string_2[:mid_point], score_matrix, indel_penalty)
    b = calculate_source_i(string_1[::-1], string_2[mid_point:][::-1], score_matrix, indel_penalty)

    # Flip the matrix of the second halves to align the sink nodes
    b = np.flip(b)
    mid_col = a[:, -1] + b[:, 0]
    mid_index = np.argmax(mid_col)
    mid = np.array([mid_index, mid_point])
    # scores.append(mid_col[mid_index])

    if mid_index + 1 < len(mid_col):
        max_val = b[mid_index, 1]
        if max_val - indel_penalty == b[mid_index, 0]:
            action = 0
        elif max_val - indel_penalty == b[mid_index + 1, 1]:
            action = 2
        else:
            action = 1
        return mid, action

    else:
        return mid, 0


def get_path_alignment(string_1, string_2, start_node, score_matrix, indel_penalty, path):
    l1 = len(string_1)
    l2 = len(string_2)
    if l1 == l2 == 0:
        return path
    if l1 == 0:
        path += [0] * l2
        # print(start_node, "*0" * l2)
        return path
    elif l2 == 0:
        # print(start_node, "*2" * l1)
        path += [2] * l1
        return path
    else:
        mid_node, mid_edge = find_middle_edge(string_1, string_2, score_matrix, indel_penalty)
        path = get_path_alignment(string_1[:mid_node[0]], string_2[:mid_node[1]], start_node, score_matrix,
                                  indel_penalty, path)
        start_node = mid_node + start_node
        # print(start_node, mid_edge)
        path.append(mid_edge)
        if mid_edge == 0:
            mid_node += [0, 1]
        elif mid_edge == 1:
            mid_node += [1, 1]
        else:
            mid_node += [1, 0]

        path = get_path_alignment(string_1[mid_node[0]:], string_2[mid_node[1]:], start_node, score_matrix,
                                  indel_penalty, path)

        return path


def linear_space_alignment(string_1, string_2, score_matrix, indel_penalty):
    # TODO Fix bug!!
    # There is a bug somewhere, need to look at it later
    path = get_path_alignment(string_1, string_2, [0, 0], score_matrix, indel_penalty, [])
    string_1 = list(string_1)
    string_2 = list(string_2)
    align_1 = ""
    align_2 = ""

    for i in path:
        if i == 0:
            align_1 = "-" + align_1
            align_2 = string_2.pop(0) + align_2
        elif i == 1:
            align_1 = string_1.pop(0) + align_1
            align_2 = string_2.pop(0) + align_2
        else:
            align_1 = string_1.pop(0) + align_1
            align_2 = "-" + align_2

    return align_1[::-1], align_2[::-1]


def affine_gap_score(string_1, string_2, match=1, miss=1, gap_open=4, gap_ext=1):
    """
    Calculate the affine gap penalty score for two strings.
    :param string_1: Aligned string
    :param string_2: Aligned string
    :param match: Score if the two characters match
    :param miss: Score if the two characters mismatch
    :param gap_open: Score if a gap opens
    :param gap_ext: Score if a gap is extended
    :return: The cumulative score of the alignment
    """
    score = 0
    p_x = False
    p_y = False
    for x, y in zip(string_1, string_2):
        if x == '-':
            if p_x:
                score -= gap_ext
            else:
                p_x = True
                score -= gap_open
            continue
        elif y == '-':
            if p_y:
                score -= gap_ext
            else:
                p_y = True
                score -= gap_open
            continue
        elif x == y:
            score += match
        else:
            score -= miss
        p_x, p_y = False, False
    return score
