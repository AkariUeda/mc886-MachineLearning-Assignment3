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
# train_set = pd.read_csv('dataset/bags.csv', header=None)
#print(train_set.head())
train_set = np.array(train_set)[:, 1:]
#print(train_set)
scaler = StandardScaler()
train_set = scaler.fit_transform(train_set, y=None)
print("Dataset loaded successfully!")


sse = []
bic_values = []
sil_values = []
calinski_values = []
davies_values = []
cluster_range = [2, 10, 30, 40, 50, 60, 70, 100,
                 200, 300, 400, 500, 700, 900, 1100, 1300, 1500, 1700, 1850, 2000]
for i in cluster_range:
    kmeans = KMeans(
        n_clusters=i, n_init=10, max_iter=1000, n_jobs=-1).fit(train_set)
    
    bic = compute_bic(kmeans, train_set)    
    cluster_labels = kmeans.predict(train_set)
    silhouette_avg = silhouette_score(train_set, cluster_labels)
    calinski_avg = calinski_harabaz_score(train_set, cluster_labels)
    davies_bouldin_avg = davies_bouldin_score(train_set, cluster_labels)
    print("For n_clusters =", i)
    print("Bic:", bic)
    print("The average silhouette_score is :", silhouette_avg)
    print("The average calinski harabaz score is:",calinski_avg)
    sse += [kmeans.inertia_]
    davies_values.append(davies_bouldin_avg)
    calinski_values.append(calinski_avg)
    bic_values.append(bic)
    sil_values.append(silhouette_avg)
    print()

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(18, 7)
fig.text(0.5, 0.03, 'Cluster Size', ha='center', va='center')
fig.text(0.06, 0.5, 'Indexes Values', ha='center', va='center', rotation='vertical')

ax1.plot(cluster_range, sse, 'r-')
ax1.set_xlabel("SSE")
ax2.plot(cluster_range, bic_values, 'b-')
ax2.set_xlabel("BIC")
plt.legend()
fig.show()
fig.savefig('exp1_elbow.png')
plt.close()

fig, (ax1, ax2, ax3)= plt.subplots(1, 3)
fig.set_size_inches(18, 7)
fig.text(0.5, 0.03, 'Cluster Size', ha='center', va='center')
fig.text(0.06, 0.5, 'Indexes Values', ha='center', va='center', rotation='vertical')

ax1.plot(cluster_range, sil_values, 'r-')
ax1.set_xlabel("Silhouette")
ax2.plot(cluster_range, calinski_values, 'g-')
ax2.set_xlabel('Calinski-Harabraz')
ax3.plot(cluster_range, davies_values, 'b-')
ax3.set_xlabel('Davies-Bouldin')

plt.legend()
fig.show()
fig.savefig('exp1_metrics.png')
plt.close()
