import numpy as np


def farthest_first_travel(data_points, k):
    data_points = np.array(data_points)
    centers = np.array([data_points[0]])

    while len(centers) < k:
        distances = np.sum((data_points - centers.reshape((len(centers), 1, -1))) ** 2, axis=2) ** 0.5
        next_cluster = np.argmax(np.min(distances, axis=0))
        centers = np.concatenate((centers, [data_points[next_cluster]]))

    return centers


def calculate_distortion(data_points, centers):
    data_points = np.array(data_points)
    centers = np.array(centers).reshape((len(centers), 1, -1))

    distances = np.sum((data_points - centers.reshape((len(centers), 1, -1))) ** 2, axis=2)
    return np.min(distances, axis=0).sum() / len(data_points)
