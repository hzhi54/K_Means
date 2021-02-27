import numpy as np
import cv2
import random
import matplotlib.pyplot as plt


def find_closest_cluster(data, initial, size):
    clusters = []
    for i in range(size):
        clusters.append([])

    for x in data:
        distance = []
        for i in range(size):
            center = initial[i]
            dist = np.linalg.norm(x - center)
            distance.append(dist)
        min_dist = np.argmin(distance)
        x = x.tolist()
        clusters[min_dist].append(x)
    return clusters


def update_initials(clusters):
    new_initials = []
    for c in clusters:
        avg_x = (np.sum(np.array(c)[:, 0]) / len(c)).round(2)
        avg_y = (np.sum(np.array(c)[:, 1]) / len(c)).round(2)
        new_initials.append([avg_x,avg_y])
    return new_initials


def randomize_initial(data, size):
    initials = []
    for s in range(size):
        rand_x = random.randint(0, len(data[0]))
        rand_y = random.randint(0, len(data[0]))
        initials.append(data[rand_x][rand_y].tolist())
    return initials


def k_means(data, initial, size, iterations):
    counter = 0
    clusters = []
    update_center = None
    while counter != iterations:
        clusters = find_closest_cluster (data, initial, size)
        initial = update_initials(clusters)
        update_center = initial
        counter += 1
    return clusters, update_center


def nfind_closest_cluster(point, initial):
    dist_idx = np.array([])
    mul = 16
    split = int(len(point)/mul)
    for i in range(mul):
        distance = np.linalg.norm(point[i*split:split+(split*i)] - initial, axis=point.ndim-1)
        temp = np.argmin(distance, axis=1)
        dist_idx = np.append(dist_idx, temp)
    return dist_idx


def nupdate_centroids(data, clusters, idx):
    points = data[clusters == idx]
    avg_point = points.mean(axis=0)
    return avg_point


def nk_means(data, initial, size, threshold):
    old_initial = 0
    cur_initial = 100
    while cur_initial - old_initial > threshold:

        for s in range(size):
            old_initial += np.sum(initial[s]) / size
        old_initial = old_initial / size

        cur_initial = 0
        clusters = nfind_closest_cluster(data[:, None, :], initial[None, :, :])
        for s in range(size):
            new_initial = nupdate_centroids(data, clusters, s)
            initial[s] = new_initial
            cur_initial += np.sum(new_initial) / size
        cur_initial = cur_initial / size

    return initial


def draw_img(img, data, centroids, K):
    clusters = nfind_closest_cluster(data[:, None, :], centroids[None, :, :])
    new_img = np.copy(data)
    for idx in range(len(clusters)):
        new_img[idx] = centroids[int(clusters[idx])]
    new_img = new_img.reshape(img.shape[0], img.shape[1], img.shape[2])
    filename = 'baboon' + str(K) + '.png'
    print("Filed saved!:", filename)
    cv2.imwrite(filename, new_img)


if __name__ == "__main__":
    X = np.array([[5.9, 3.2], [4.6, 2.9], [6.2, 2.8],
                  [4.7, 3.2], [5.5, 4.2], [5.0, 3.0],
                  [4.9, 3.1], [6.7, 3.1], [5.1, 3.8],
                  [6.0, 3.0]])

    x_axis = X[:, 0]
    y_axis = X[:, 1]

    centers = np.array([[6.2, 3.2], [6.6, 3.7], [6.5, 3.0]])
    labels = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
    N = 10
    K = len(centers)

    clusters = find_closest_cluster(X, centers, K)
    plt.figure(0)
    plt.scatter(np.array(clusters[0])[:, 0], np.array(clusters[0])[:, 1], marker='^', c='red')
    plt.scatter(np.array(clusters[1])[:, 0], np.array(clusters[1])[:, 1], marker='^', c='blue')
    plt.scatter(np.array(clusters[2])[:, 0], np.array(clusters[2])[:, 1], marker='^', c='green')
    for c in clusters:
        for point in c:
            plt.annotate("("+str(point[0])+","+str(point[1])+")", (point[0], point[1]))
    plt.savefig('task2_iter1_a.jpg')

    clusters, update_center = k_means(X, centers, K, N)
    plt.figure(1)
    plt.scatter(np.array(clusters[0])[:, 0], np.array(clusters[0])[:, 1], marker='^', facecolors='none', edgecolors='c')
    plt.scatter(update_center[0][0], update_center[0][1], marker='o', c='red')
    plt.scatter(np.array(clusters[1])[:, 0], np.array(clusters[1])[:, 1], marker='^', facecolors='none', edgecolors='c')
    plt.scatter(update_center[1][0], update_center[1][1], marker='o', c='blue')
    plt.scatter(np.array(clusters[2])[:, 0], np.array(clusters[2])[:, 1], marker='^', facecolors='none', edgecolors='c')
    plt.scatter(update_center[2][0], update_center[2][1], marker='o', c='green')
    for c in clusters:
        for point in c:
            plt.annotate("("+str(point[0])+","+str(point[1])+")", (point[0], point[1]))
    for u in update_center:
        plt.annotate("("+str(u[0])+","+str(u[1])+")", (u[0], u[1]))
    plt.savefig('task2_iter1_b.jpg')

    threshold = 0.1
    img = cv2.imread('baboon.png')
    data = img.reshape(len(img)*len(img[0]), 3)
    K_list = [3, 5, 10, 20]
    for K in K_list:
        init = data[random.sample(range(data.shape[0]), K)]
        centroids = nk_means(data, init, K, threshold)
        draw_img(img, data, centroids, K)
