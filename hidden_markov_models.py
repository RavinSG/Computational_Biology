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
        return round(num, 3)


def print_profiles(transitions, counts, alphabet):
    mapping = {j: i + 1 for i, j in enumerate(alphabet.split(" "))}
    states = ["M", "D", "I"]
    col_order = ["I", "M", "D"]
    addition = [0, 1, 1]
    header = ["", "S", "I0"]

    transition_matrix = []
    count_row = ['0'] * len(mapping)
    count_matrix = [["S"] + count_row]
    first_row = [0] + [transitions[0]["M0"]["I0"], transitions[0]["M0"]["M1"], transitions[0]["M0"]["D1"]] + [0] * (
            3 * (len(transitions) - 1) - 1)
    second_row = [0] + [transitions[0]["I0"]["I0"], transitions[0]["I0"]["M1"], transitions[0]["I0"]["D1"]] + [0] * (
            3 * (len(transitions) - 1) - 1)
    second_count = ["I0"] + count_row
    for i, j in counts["I0"].items():
        second_count[mapping[i]] = str(j)
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
            count_matrix.append(row_count)
            for next_state, idx in zip(col_order, addition):
                trans_state = f'{next_state}{i + idx}'
                vector.append(format_number(transitions[i][current_state][trans_state]))
            vector += [0] * (3 * (len(transitions) - i) - 3)
            transition_matrix.append(vector[:-1])

    transition_matrix.append([0] * len(transition_matrix[0]))
    header.append("E")
    count_matrix.append(["E"] + count_row)
    print("\t".join(header))

    for j, i in enumerate(transition_matrix):
        print("\t".join([header[j + 1]] + list(map(str, i))))
    print('--------')

    print("\t".join([""] + alphabet.split(" ")))
    for i in count_matrix:
        print("\t".join(i))
