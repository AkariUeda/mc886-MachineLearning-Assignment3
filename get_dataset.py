import os
import sys
import sklearn
import pandas as pd
import numpy as np

data_file = sys.argv[1]
original_tweets = sys.argv[2]

dataset = pd.read_csv(data_file, header=None, sep=',')

dataset = dataset.sample(frac=1)

train_set = dataset.iloc[:int(len(dataset)*0.85), :]

test_set = dataset.iloc[int(len(dataset)*0.85):, :]

train_set.to_csv('train_set.csv')
test_set.to_csv('test_set.csv')

print("New train and test sets saved successfully!")

