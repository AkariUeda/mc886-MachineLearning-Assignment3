import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance

clusters = int(sys.argv[1])
n_medoids = int(sys.argv[2])
n_random = int(sys.argv[3])
datasetfile = sys.argv[4]

train_set = pd.read_csv('train_set.csv')

tweets = pd.read_csv('./dataset/health.txt', sep='|')

test_tweets =  np.array(train_set, np.int32)[:,0]

train_set = np.array(train_set)[:,1:]

scaler = StandardScaler()
train_set = scaler.fit_transform(train_set, y=None)
pca = PCA(n_components=0.90, svd_solver='full')
train_set=pca.fit_transform(train_set)
print(train_set.shape)

clustering = KMeans(n_clusters=clusters, n_init=10, max_iter=10000, n_jobs=-1).fit(train_set)
# number_of_clusters = max(labels)+1

# clustering = DBSCAN(eps=0.5, min_samples=5).fit(train_set)
# labels = clustering.labels_

# clustering = SpectralClustering(n_clusters=clusters).fit(train_set)
labels = clustering.labels_
number_of_clusters = max(labels)+1

idx_clusters = np.array([np.where(labels==i) for i in range(number_of_clusters)])
clusters = np.array([train_set[np.where(labels==i)] for i in range(number_of_clusters)])

for c in range(number_of_clusters):

	dist_matrix = distance.cdist(clusters[c], clusters[c])
	medoid = np.argmin(np.sum(dist_matrix,axis=0))

	if n_medoids < len(dist_matrix[medoid]):
		n_neighbors = np.argpartition(dist_matrix[medoid],n_medoids)[:n_medoids]
	else:
		n_neighbors = np.argpartition(dist_matrix[medoid],len(dist_matrix[medoid])-1)

	print("\nMedoids cluster {} (size={}): {}".format(c,len(clusters[c]),tweets.iloc[test_tweets[idx_clusters[c][0][medoid]]]['headline_text']))
	print("   {} nearest points:".format(len(n_neighbors)))
	for n in range(len(n_neighbors)):
		print("          {}".format(tweets.iloc[test_tweets[idx_clusters[c][0][n_neighbors[n]]]]['headline_text']))
	
	print("   {} random points:".format(n_random))
	random_idx = np.random.randint(len(idx_clusters[c][0]),size=n_random)

	for n in range(n_random):
		print("          {}".format(tweets.iloc[test_tweets[idx_clusters[c][0][random_idx[n]]]]['headline_text']))

