'''

Find the K centroids in the train set, print the N nearest medoids' neighbors
and N random clusters' points.

Parameters:

clusters: number of clusters to train
n_medoids: N nearest medoids' neighbors to print
n_random: N random cluters' points to print

'''

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
import sys
from exp3_evaluation_metrics import evaluate
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance

clusters = int(sys.argv[1])
n_medoids = int(sys.argv[2])
n_random = int(sys.argv[3])

train_set = pd.read_csv('train_set.csv')

test_set = pd.read_csv('test_set.csv')

tweets_text = pd.read_csv('./dataset/health.txt', sep='|')

train_tweets =  np.array(test_set, np.int32)[:,0]

test_set = np.array(test_set)[:,1:]

train_set = np.array(train_set)[:,1:]

clustering = KMeans(n_clusters=clusters, n_init=10, max_iter=10000, n_jobs=-1).fit(test_set)
labels = clustering.labels_
number_of_clusters = max(labels)+1

idx_clusters = np.array([np.where(labels==i) for i in range(number_of_clusters)])
clusters = np.array([test_set[np.where(labels==i)] for i in range(number_of_clusters)])

metrics = evaluate(np.copy(train_set), np.copy(test_set), algorithm='kmeans', normalize=False, use_pca=True, pca_variance=0.9)

for c in range(number_of_clusters):

	dist_matrix = distance.cdist(clusters[c], clusters[c])
	medoid = np.argmin(np.sum(dist_matrix,axis=0))

	if n_medoids < len(dist_matrix[medoid]):
		n_neighbors = np.argpartition(dist_matrix[medoid],n_medoids)[:n_medoids]
	else:
		n_neighbors = np.argpartition(dist_matrix[medoid],len(dist_matrix[medoid])-1)

	print("\nMedoids cluster {} (size={}): {}".format(c,len(clusters[c]),tweets_text.iloc[train_tweets[idx_clusters[c][0][medoid]]]['headline_text']))
	print("   {} nearest points:".format(len(n_neighbors)))
	for n in range(len(n_neighbors)):
		print("          {}".format(tweets_text.iloc[train_tweets[idx_clusters[c][0][n_neighbors[n]]]]['headline_text']))
	
	print("   {} random points:".format(n_random))
	random_idx = np.random.randint(len(idx_clusters[c][0]),size=n_random)

	for n in range(n_random):
		print("          {}".format(tweets_text.iloc[train_tweets[idx_clusters[c][0][random_idx[n]]]]['headline_text']))

