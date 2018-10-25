import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import sys
from sklearn.preprocessing import StandardScaler




train_set = pd.read_csv('train_set.csv')
test_set = pd.read_csv('test_set.csv')

train_set = np.array(train_set)[:,1:]
test_set = np.array(test_set)[:,1:]

scaler = StandardScaler()
train_set = scaler.fit_transform(train_set, y=None)

variances = [0.99, 0.95, 0.9]

for v in variances:
	pca = PCA(n_components=v, svd_solver='full')
	new_features = pca.fit_transform(train_set)
