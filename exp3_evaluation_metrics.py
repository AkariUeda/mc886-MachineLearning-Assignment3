from __future__ import print_function

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabaz_score, davies_bouldin_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import sys
import pandas as pd
import numpy as np
from sklearn import cluster
from scipy.spatial import distance
import sklearn.datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sys.stderr = open('/tmp/tt', 'w') # FIXME

def evaluate(train_set, valid_set, algorithm='kmeans', normalize=False, use_pca=False, pca_variance=0.99):

    print(f'algorithm={algorithm} normalize={normalize} use_pca={use_pca} pca_variance={pca_variance}')

    if normalize:
        scaler = StandardScaler()
        scaler.fit(train_set)
        train_set = scaler.transform(train_set)
        valid_set = scaler.transform(valid_set)

    if use_pca:
        pca = PCA(n_components=pca_variance, svd_solver='full')
        pca.fit(train_set)
        print(f'PCA components for maintaining {pca_variance} variance: {pca.n_components_}')
        train_set = pca.transform(train_set)
        valid_set = pca.transform(valid_set)

    if algorithm == 'kmeans':
        clustering = KMeans(n_clusters=60, n_init=10, max_iter=10000, n_jobs=-1).fit(train_set)

    validation_labels = clustering.predict(valid_set)

    silhouette_avg = silhouette_score(valid_set, validation_labels)
    davies_bouldin_avg = davies_bouldin_score(valid_set, validation_labels)
    calinski_avg = calinski_harabaz_score(valid_set, validation_labels)

    print(f"silhouette_score:{silhouette_avg}, davies_bouldin_score:{davies_bouldin_avg}, calinski_harabaz_score={calinski_avg}")
    print()

    return (silhouette_avg, davies_bouldin_avg, calinski_avg)

if __name__ == '__main__':
            
    print("Reading dataset...")
    train_set = pd.read_csv('train_set.csv')
    train_set = np.array(train_set)[:, 1:]
    valid_set = pd.read_csv('valid_set.csv')
    valid_set = np.array(valid_set)[:, 1:]
    print("Dataset loaded successfully!")


    num_experiments = 10
    variance_values = [0.9, 0.95, 0.99]
    results_silhouette = np.zeros((num_experiments, 2, len(variance_values) + 1))
    results_davies = np.zeros((num_experiments, 2, len(variance_values) + 1))
    results_calinski = np.zeros((num_experiments, 2, len(variance_values) + 1))

    for i in range(num_experiments):
        for should_normalize in [False, True]:
            for j in range(len(variance_values)):
                (results_silhouette[i][int(should_normalize)][j], results_davies[i][int(should_normalize)][j], results_calinski[i][int(should_normalize)][j]) = evaluate(np.copy(train_set), np.copy(valid_set), algorithm='kmeans', normalize=should_normalize, use_pca=True, pca_variance=variance_values[j])
            (results_silhouette[i][int(should_normalize)][len(variance_values)], results_davies[i][int(should_normalize)][len(variance_values)], results_calinski[i][int(should_normalize)][len(variance_values)]) = evaluate(np.copy(train_set), np.copy(valid_set), algorithm='kmeans', normalize=should_normalize, use_pca=False, pca_variance=0.99)

    """
    print(np.average(results_silhouette, axis=0))
    print(np.average(results_davies, axis=0))
    print(np.average(results_calinski, axis=0))

    print(np.std(results_silhouette, axis=0))
    print(np.std(results_davies, axis=0))
    print(np.std(results_calinski, axis=0))
    """

    from uncertainties import unumpy

    print(unumpy.uarray(np.average(results_silhouette, axis=0), np.std(results_silhouette, axis=0)))
    print()
    print(unumpy.uarray(np.average(results_davies, axis=0), np.std(results_davies, axis=0)))
    print()
    print(unumpy.uarray(np.average(results_calinski, axis=0), np.std(results_calinski, axis=0)))
    print()
