from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import sys
import pandas as pd
import matplotlib.pyplot as plt


train_set = pd.read_csv('train_set.csv')

test_set = pd.read_csv('test_set.csv')


sse = []

for i in range(2, 21):
    kmeans = KMeans(
        n_clusters=i, n_init=10, max_iter=1000, n_jobs=-1).fit(train_set)
    print(kmeans.inertia_)
    cluster_labels = kmeans.predict(train_set)
    silhouette_avg = silhouette_score(train_set, cluster_labels)
    print("For n_clusters =", i,
          "The average silhouette_score is :", silhouette_avg)
    sse += [kmeans.inertia_]


plt.plot(range(2, 21), sse, 'r-')


plt.show()
