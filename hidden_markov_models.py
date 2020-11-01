import numpy as np
from collections import defaultdict, Counter


def hidden_path_prob(hidden_path, transition_prob):
    prob = 0.5
    for i, j in zip(hidden_path, hidden_path[1:]):
        prob *= transition_prob[i][j]

    return prob


def outcome_prob(string, hidden_path, emission_prob):
    prob = 1
    for char, state in zip(string, hidden_path):
        prob *= emission_prob[state][char]

    return prob


def viterbi_algorithm(string, alphabet, states, transition_prob, emission_prob):
    alphabet = {alphabet[i]: i for i in range(len(alphabet))}
    num_states = len(states)
    start_prob = np.ones((num_states, 1))
    backtrack = np.zeros((num_states, 1)) - 1

    for i in range(len(string)):
        state_values = start_prob * emission_prob[:, alphabet[string[i]]].reshape(num_states, 1)
        next_states = state_values * transition_prob
        next_backtrack = np.argmax(next_states, axis=1)
        backtrack = np.hstack((backtrack, next_backtrack.reshape(num_states, 1)))
        start_prob = next_states[np.arange(num_states), [next_backtrack]]

    backtrack = backtrack.astype(int)
    state_transition = ""
    state = np.argmax(start_prob)
    while state != -1:
        state_transition = states[state] + state_transition
        state = backtrack[:, -1][state]
        backtrack = backtrack[:, :-1]

    return state_transition[1:]


def outcome_likelihood(string, alphabet, states, transition_prob, emission_prob):
    alphabet = {alphabet[i]: i for i in range(len(alphabet))}
    num_states = len(states)
    start_prob = np.ones((num_states, 1)) / num_states
    for i in range(len(string)):
        state_values = start_prob * emission_prob[:, alphabet[string[i]]].reshape(num_states, 1)
        next_states = state_values * transition_prob
        start_prob = np.sum(next_states, axis=0).reshape(num_states, 1)

    return np.sum(start_prob)


def generate_hmm_profile(threshold, alphabet, alignments):
    alignments = np.array([list('A' + x + 'A') for x in alignments])
    num_seq, align_len = alignments.shape
    transitions = dict()
    counts = defaultdict(lambda: defaultdict(float))

    keep_cols = []
    for i in range(align_len):
        removal_prob = np.count_nonzero(alignments[:, i] == '-') / num_seq
        keep_cols.append(removal_prob < threshold)

    i = 1
    col_num = 0
    while i < align_len:
        column_transitions = defaultdict(lambda: defaultdict(int))
        comp_cols = alignments[:, i - 1:i + 1]
        indels = np.count_nonzero(comp_cols[:, 0] == '-')

        pct = 1 / (num_seq - indels)
        del_pct = 1 / max(indels, 1)
        cur_rem = not keep_cols[i]

        del_cols = None
        next_cols = None
        if cur_rem:
            column_nucleotides = defaultdict(float)
            j = i
            while not keep_cols[j]:
                j += 1

            next_cols = alignments[:, i:j + 1]
            del_cols = next_cols[:, :-1]
            nucleotides = del_cols[del_cols != '-']
            counter = Counter(nucleotides)

            for nuc, count in counter.items():
                column_nucleotides[nuc] = count / len(nucleotides)
            counts[f"I{col_num}"] = column_nucleotides

            del_cols = np.sum((del_cols != '-').astype(int), axis=1)
            nuc_num = np.sum(del_cols)
            self_transition = del_cols - 1
            self_transition[self_transition < 0] = 0
            self_transition = np.sum(self_transition)
            deletions = np.count_nonzero((next_cols[:, -1] != '-').astype(int) - (del_cols > 0) < 0)
            matches = nuc_num - self_transition - deletions

            column_transitions[f'I{col_num}'][f'I{col_num}'] = self_transition / nuc_num
            column_transitions[f'I{col_num}'][f'M{col_num + 1}'] = matches / nuc_num
            column_transitions[f'I{col_num}'][f'D{col_num + 1}'] = deletions / nuc_num

            i = j

        column_nucleotides = defaultdict(float)
        first_col = comp_cols[:, 0]
        nucleotides = first_col[first_col != '-']
        counter = Counter(nucleotides)
        for nuc, count in counter.items():
            column_nucleotides[nuc] = count / len(nucleotides)
        counts[f"M{col_num}"] = column_nucleotides

        if not cur_rem:
            for j in comp_cols:
                if j[0] != '-' and j[1] != '-':
                    column_transitions[f'M{col_num}'][f'M{col_num + 1}'] += pct
                elif j[0] != '-':
                    column_transitions[f'M{col_num}'][f'D{col_num + 1}'] += pct
                elif j[0] == '-' and j[1] == '-':
                    column_transitions[f'D{col_num}'][f'D{col_num + 1}'] += del_pct
                else:
                    column_transitions[f'D{col_num}'][f'M{col_num + 1}'] += del_pct
        else:
            del_cols = (del_cols > 0).astype(int)
            for k, j in enumerate(comp_cols):
                if j[0] != '-':
                    if del_cols[k] == 1:
                        column_transitions[f'M{col_num}'][f'I{col_num}'] += pct
                    elif next_cols[k, -1] != '-':
                        column_transitions[f'M{col_num}'][f'M{col_num + 1}'] += pct
                    else:
                        column_transitions[f'M{col_num}'][f'D{col_num + 1}'] += pct

                else:
                    if del_cols[k] == 1:
                        column_transitions[f'D{col_num}'][f'I{col_num}'] += del_pct
                    elif next_cols[k, -1] != '-':
                        column_transitions[f'D{col_num}'][f'M{col_num + 1}'] += del_pct
                    else:
                        column_transitions[f'D{col_num}'][f'D{col_num + 1}'] += del_pct

        transitions[col_num] = column_transitions
        i += 1
        col_num += 1
    return transitions, counts


def format_number(num):
    if num == 0:
        return int(num)
    else:
        return num


def add_pseudo_counts_transition(dict_row, pseudo_count, row_num, last_row):
    col_order = ["I", "M", "D"]
    addition = [0, 1, 1]
    tot_prob = round(sum(dict_row.values())) + 3 * pseudo_count

    if last_row:
        col_order = col_order[:2]
        addition = addition[:2]
        tot_prob = round(sum(dict_row.values())) + 2 * pseudo_count

    for j, i in zip(col_order, addition):
        next_state = f"{j}{i + row_num}"
        dict_row[next_state] = (dict_row[next_state] + pseudo_count) / tot_prob


def add_pseudo_counts_nucleotide(counts, pseudo_count):
    prob = np.array(counts[1:]).astype(float)
    prob = (prob + pseudo_count)

    return [counts[0]] + (prob / sum(prob)).astype(str).tolist()


def update_profile(transitions, counts, alphabet, pseudo_count=0.01, print_profile=False):
    mapping = {j: i + 1 for i, j in enumerate(alphabet.split(" "))}
    states = ["M", "D", "I"]
    col_order = ["I", "M", "D"]
    addition = [0, 1, 1]
    header = ["", "S", "I0"]

    transition_matrix = []
    count_row = ['0'] * len(mapping)
    count_matrix = [["S"] + count_row]
    add_pseudo_counts_transition(transitions[0]["M0"], pseudo_count, 0, False)
    add_pseudo_counts_transition(transitions[0]["I0"], pseudo_count, 0, False)
    first_row = [0] + [transitions[0]["M0"]["I0"], transitions[0]["M0"]["M1"],
                       transitions[0]["M0"]["D1"]] + [0] * (3 * (len(transitions) - 1) - 1)
    second_row = [0] + [transitions[0]["I0"]["I0"], transitions[0]["I0"]["M1"],
                        transitions[0]["I0"]["D1"]] + [0] * (3 * (len(transitions) - 1) - 1)
    second_count = ["I0"] + count_row

    for i, j in counts["I0"].items():
        second_count[mapping[i]] = str(j)
    second_count = add_pseudo_counts_nucleotide(second_count, pseudo_count)
    count_matrix.append(second_count)

    transition_matrix += [first_row, second_row]
    for i in range(1, len(transitions)):
        for state in states:
            vector = [0] * (1 + 3 * i)
            current_state = f'{state}{i}'
            header.append(current_state)
            row_count = [current_state] + count_row
            for k, j in counts[current_state].items():
                row_count[mapping[k]] = str(format_number(j))
            if state != 'D':
                row_count = add_pseudo_counts_nucleotide(row_count, pseudo_count)
            count_matrix.append(row_count)
            add_pseudo_counts_transition(transitions[i][current_state], pseudo_count, i,
                                         False if i != len(transitions) - 1 else True)
            for next_state, idx in zip(col_order, addition):
                trans_state = f'{next_state}{i + idx}'
                vector.append(format_number(transitions[i][current_state][trans_state]))
            vector += [0] * (3 * (len(transitions) - i) - 3)
            transition_matrix.append(vector[:-1])

    transition_matrix.append([0] * len(transition_matrix[0]))
    header.append("E")
    count_matrix.append(["E"] + count_row)
    updated_count = defaultdict(lambda: defaultdict(float))

    for i in count_matrix:
        updated_count[i[0]] = defaultdict(float, {k: float(j) for k, j in zip(alphabet.split(" "), i[1:])})
    if print_profile:
        print("\t".join(header))

        for j, i in enumerate(transition_matrix):
            print("\t".join([header[j + 1]] + list(map(str, i))))
        print('--------')

        print("\t".join([""] + alphabet.split(" ")))
        for i in count_matrix:
            print("\t".join(i))

    return updated_count


def align_sequence_to_profile(sequence, profile, nuc_prob):
    sequence = sequence + "$"
    num_cols = len(sequence)
    num_rows = 3 * len(profile)
    graph = np.zeros((num_rows, num_cols))

    graph[0, 0] = 1
    graph[0, 1] = 1 * profile[0]["M0"]["I0"] * nuc_prob["I0"][sequence[0]]
    graph[1, 1] = 1 * profile[0]["M0"]["M1"] * nuc_prob["M1"][sequence[0]]
    graph[2, 0] = 1 * profile[0]["M0"]["D1"]
    backtrack = np.zeros((num_rows, num_cols, 2))

    for i in range(1, len(profile) - 1):
        col = 2 + 3 * (i - 1)
        graph[col + 3, 0] = graph[col, 0] * profile[i][f"D{i}"][f"D{i + 1}"]
        graph[col + 1, 1] = graph[col, 0] * profile[i][f"D{i}"][f"I{i}"] * nuc_prob[f"I{i}"][sequence[i]]
        graph[col + 2, 1] = graph[col, 0] * profile[i][f"D{i}"][f"M{i + 1}"] * nuc_prob[f"M{i + 1}"][sequence[i]]
        backtrack[col + 3, 0] = [col, 0]
        backtrack[col + 1, 1] = [col, 0]
        backtrack[col + 2, 1] = [col, 0]

    for i in range(1, num_cols - 1):
        for j in range(num_rows - 2):
            level_num = j // 3
            if j % 3 == 0:
                val_1 = graph[j, i] * profile[level_num][f"I{level_num}"][f"I{level_num}"] * \
                        nuc_prob[f"I{level_num}"][sequence[i]]
                val_2 = graph[j, i] * profile[level_num][f"I{level_num}"][f"M{level_num + 1}"] * \
                        nuc_prob[f"M{level_num + 1}"][sequence[i]]
                val_3 = graph[j, i] * profile[level_num][f"I{level_num}"][f"D{level_num + 1}"]

                if val_1 > graph[j, i + 1]:
                    graph[j, i + 1] = val_1
                    backtrack[j, i + 1] = [j, i]

                if val_2 > graph[j + 1, i + 1]:
                    graph[j + 1, i + 1] = val_2
                    backtrack[j + 1, i + 1] = [j, i]

                if val_3 > graph[j + 2, i]:
                    graph[j + 2, i] = val_3
                    backtrack[j + 2, i] = [j, i]

            elif j % 3 == 1:
                level_num += 1
                val_1 = graph[j, i] * profile[level_num][f"M{level_num}"][f"I{level_num}"] * \
                        nuc_prob[f"I{level_num}"][sequence[i]]
                val_2 = graph[j, i] * profile[level_num][f"M{level_num}"][f"M{level_num + 1}"] * \
                        nuc_prob[f"M{level_num + 1}"][sequence[i]]
                val_3 = graph[j, i] * profile[level_num][f"M{level_num}"][f"D{level_num + 1}"]

                if val_1 > graph[j + 2, i + 1]:
                    graph[j + 2, i + 1] = val_1
                    backtrack[j + 2, i + 1] = [j, i]

                if val_2 > graph[j + 3, i + 1]:
                    graph[j + 3, i + 1] = val_2
                    backtrack[j + 3, i + 1] = [j, i]

                if val_3 > graph[j + 4, i]:
                    graph[j + 4, i] = val_3
                    backtrack[j + 4, i] = [j, i]
            else:
                level_num += 1
                val_1 = graph[j, i] * profile[level_num][f"D{level_num}"][f"I{level_num}"] * \
                        nuc_prob[f"I{level_num}"][sequence[i]]
                val_2 = graph[j, i] * profile[level_num][f"D{level_num}"][f"M{level_num + 1}"] * \
                        nuc_prob[f"M{level_num + 1}"][sequence[i]]
                val_3 = graph[j, i] * profile[level_num][f"D{level_num}"][f"D{level_num + 1}"]

                if val_1 > graph[j + 1, i + 1]:
                    graph[j + 1, i + 1] = val_1
                    backtrack[j + 1, i + 1] = [j, i]
                if val_2 > graph[j + 2, i + 1]:
                    graph[j + 2, i + 1] = val_2
                    backtrack[j + 2, i + 1] = [j, i]
                if val_3 > graph[j + 3, i]:
                    graph[j + 3, i] = val_3
                    backtrack[j + 3, i] = [j, i]

    # Should the node with the highest value or the final match node must be used ?
    # end_node = [num_rows - 5 + (np.argmax(graph[:, -1][-5:-2])), num_cols - 1]
    end_node = [num_rows - 5, num_cols - 1]

    states = ["I", "M", "D"]
    path = []

    while not np.array_equal(end_node, [0, 0]):
        level = end_node[0] // 3
        state = end_node[0] % 3
        if state != 0:
            level += 1
        path.append(f'{states[state]}{level}')
        end_node = backtrack[end_node[0], end_node[1]].astype(int)

    path.reverse()
    return " ".join(path)
