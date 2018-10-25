import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import sys
from scipy.spatial import distance

clusters = int(sys.argv[1])
n_medoids = int(sys.argv[2])
n_random = int(sys.argv[3])
train_set = pd.read_csv('train_set.csv')
test_set = pd.read_csv('test_set.csv')
tweets = pd.read_csv('./dataset/health.txt', sep='|')

train_tweets = np.array(train_set, np.int32)[:,0]
test_tweets =  np.array(test_set, np.int32)[:,0]

train_set = np.array(train_set)[:,1:]
test_set = np.array(test_set)[:,1:]

kmeans = KMeans(n_clusters=clusters, n_init=10, max_iter=1000, n_jobs=-1).fit(train_set)

labels = kmeans.predict(test_set)
number_of_clusters = max(labels)+1

idx_clusters = np.array([np.where(labels==i) for i in range(number_of_clusters)])
clusters = np.array([test_set[np.where(labels==i)] for i in range(number_of_clusters)])

for c in range(number_of_clusters):
	dist_matrix = distance.cdist(clusters[c], clusters[c])
	medoid = np.argmin(np.sum(dist_matrix,axis=0))
	n_neighbors = np.argpartition(dist_matrix[medoid],n_medoids+1)[:n_medoids+1]

	print("\nMedoids cluster {}: {}".format(c,tweets.iloc[test_tweets[idx_clusters[c][0][medoid]]]['headline_text']))
	print("   {} nearest points:".format(n_medoids))
	for n in range(len(n_neighbors)):
		print("          {}".format(tweets.iloc[test_tweets[idx_clusters[c][0][n_neighbors[n]]]]['headline_text']))
	
	print("   {} random points:".format(n_random))
	random_idx = np.random.randint(len(idx_clusters[c][0]),size=n_random)

	for n in range(n_random):
		print("          {}".format(tweets.iloc[test_tweets[idx_clusters[c][0][random_idx[n]]]]['headline_text']))

