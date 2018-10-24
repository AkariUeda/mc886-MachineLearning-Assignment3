import sklearn.cluster
import sys
import pandas as pd
import matplotlib.pyplot as plt


train_set = pd.read_csv('train_set.csv')

test_set = pd.read_csv('test_set.csv')


sse = []

for i in range(1, 21):
    kmeans = sklearn.cluster.KMeans(
        n_clusters=i, n_init=10, max_iter=1000, n_jobs=-1).fit(train_set)
    print(kmeans.inertia_)
    sse += [kmeans.inertia_]


plt.plot(range(1, 21), sse, 'r-')


plt.show()
