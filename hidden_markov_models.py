import numpy as np


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
