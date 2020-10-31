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
