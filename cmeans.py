import warnings
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


# Συνάρτηση kmeans
def kmeans(x, k, no_of_iterations):

    idx = np.random.choice(len(x), k, replace=False)
    # Τυχαία επιλογή κεντροειδών
    centroids = x[idx, :]

    # Εύρεση της ευκλείδειας απόστασης μεταξύ των κεντροειδών
    # και για κάθε σημείο στο σετ μας
    distances = cdist(x, centroids, 'euclidean')

    # Κεντροειδή με την ελάχιστη απόσταση
    points = np.array([np.argmin(i) for i in distances])

    # Επανάληψη των προηγούμενων βημάτων για συγκεκριμένο
    # αριθμό επαναλήψεων
    for _ in range(no_of_iterations):
        centroids = []
        for idx in range(k):
            # Ενημέρωση κεντροειδών λαμβάνοντας τον μέσο όρο
            # από την εκάστοτε συστάδα
            temp_cent = x[points == idx].mean(axis=0)
            centroids.append(temp_cent)

        centroids = np.vstack(centroids)  # Ενημέρωση κεντροειδών

        distances = cdist(x, centroids, 'euclidean')
        points = np.array([np.argmin(i) for i in distances])

    return points


def input_data(data, name):

    pca = PCA(2)

    # Μεταμόρφωση δεδομένων
    df = pca.fit_transform(data)

    # Κλήση συνάρτησης και θέτουμε το δεύτερο
    # όρισμα με 3 εφόσον θέλουμε 3 συστάδες.
    label = kmeans(df, 3, 1000)

    # Απεικόνιση των αποτελεσμάτων
    u_labels = np.unique(label)
    for i in u_labels:
        plt.scatter(df[label == i, 0], df[label == i, 1], label=i)

    plt.title(name)
    plt.legend()
    plt.show()

    for i in u_labels:
        plt.bar(df[label == i, 0], df[label == i, 1], label=i)

    plt.title(name)
    plt.legend()
    plt.show()
