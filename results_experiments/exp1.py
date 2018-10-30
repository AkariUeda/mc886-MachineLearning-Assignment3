from __future__ import print_function

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

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

# Cite: https://pdfs.semanticscholar.org/1dc9/549f55ab95bc4b8a3f10371ff7956c6c9411.pdf

def compute_bic(kmeans, X):
    """
    Computes the BIC metric for a given clusters

    Parameters:
    -----------------------------------------
    kmeans:  List of clustering object from scikit learn

    X     :  multidimension np array of data points

    Returns:
    -----------------------------------------
    BIC value
    """
    # print(X)
    # assign centers and labels
    centers = [kmeans.cluster_centers_]
    labels = kmeans.labels_
    # number of clusters
    m = kmeans.n_clusters
    # size of the clusters
    n = np.bincount(labels)
    # size of data set
    N, d = X.shape

    # compute variance for all clusters beforehand
    cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]],
                                                           'euclidean')**2) for i in range(m)])

    const_term = 0.5 * m * np.log(N) * (d+1)

    BIC = np.sum([n[i] * np.log(n[i]) -
                  n[i] * np.log(N) -
                  ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
                  ((n[i] - 1) * d / 2) for i in range(m)]) - const_term

    return(BIC)


print("Reading dataset...")
train_set = pd.read_csv('train_set.csv')
print(train_set.head())
train_set = np.array(train_set)[:, 1:]
print(train_set)
print("Dataset loaded successfully!")


sse = []
bic_values = []
cluster_range = range(2, 201, 1)
for i in cluster_range:
    kmeans = KMeans(
        n_clusters=i, n_init=10, max_iter=10000, n_jobs=-1).fit(train_set)

    bic = compute_bic(kmeans, train_set)
    print("For n_clusters =", i)
    print("BIC:", bic)
    print("SSE:", kmeans.inertia_)
    sse += [kmeans.inertia_]
    bic_values.append(bic)
    print()

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(18, 7)
fig.text(0.5, 0.03, '', ha='center', va='center')
fig.text(0.06, 0.5, '', ha='center',
         va='center', rotation='vertical')

ax1.plot(cluster_range, sse, 'r-')
ax1.set_xlabel("Cluster Size")
ax1.set_ylabel("SSE")
ax2.plot(cluster_range, bic_values, 'b-')
ax2.set_xlabel("Cluster Size")
ax2.set_ylabel("BIC")
plt.legend()
#fig.savefig('exp1_elbow.png')
plt.show()
