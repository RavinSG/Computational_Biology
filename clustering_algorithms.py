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


def k_means_clustering(data_points, k, random_start=True):
    data_points = np.array(data_points)
    if random_start:
        while True:
            centers = data_points[np.random.choice(len(data_points), k)]
            if len(set(map(tuple, centers))) == k:
                centers = centers.reshape(k, 1, -1)
                break
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
