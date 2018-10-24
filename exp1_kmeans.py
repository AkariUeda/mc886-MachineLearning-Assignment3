import sklearn.cluster
import sys
import pandas as pd
import matplotlib.pyplot as plt


train_set = pd.read_csv('train_set.csv')

test_set = pd.read_csv('test_set.csv')


sse = []

for i in range(1,11):
	kmeans = sklearn.cluster.KMeans(n_clusters=i, max_iter=1000).fit(train_set)
	print(kmeans.inertia_)
	sse += [kmeans.inertia_]


plt.plot(range(1,11), sse,'r-')


plt.show()