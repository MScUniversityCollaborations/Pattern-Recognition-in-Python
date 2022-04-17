import warnings
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


# Defining our function
def kmeans(x, k, no_of_iterations):
    idx = np.random.choice(len(x), k, replace=False)
    # Randomly choosing Centroids
    centroids = x[idx, :]  # Step 1

    # finding the distance between centroids and all the data points
    distances = cdist(x, centroids, 'euclidean')  # Step 2

    # Centroid with the minimum Distance
    points = np.array([np.argmin(i) for i in distances])  # Step 3

    # Repeating the above steps for a defined number of iterations
    # Step 4
    for _ in range(no_of_iterations):
        centroids = []
        for idx in range(k):
            # Updating Centroids by taking mean of Cluster it belongs to
            temp_cent = x[points == idx].mean(axis=0)
            centroids.append(temp_cent)

        centroids = np.vstack(centroids)  # Updated Centroids

        distances = cdist(x, centroids, 'euclidean')
        points = np.array([np.argmin(i) for i in distances])

    return points


def start(data, name):
    # print(data)
    pca = PCA(2)

    # Transform the data
    df = pca.fit_transform(data)

    # Applying our function
    label = kmeans(df, 3, 1000)

    # Visualize the results
    u_labels = np.unique(label)
    for i in u_labels:
        plt.scatter(df[label == i, 0], df[label == i, 1], label=i)
        # plt.bar(df[label == i, 0], df[label == i, 1], label=i)

    plt.title(name)
    plt.legend()
    plt.show()

    for i in u_labels:
        plt.bar(df[label == i, 0], df[label == i, 1], label=i)

    plt.title(name)
    plt.legend()
    plt.show()