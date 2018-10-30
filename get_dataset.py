'''
How to use it:

python3 get_dataset.py [path to dataset file (e.g., dataset/bags.csv)] [path to tweets file (e.g., dataset/health.txt)]

Warning: This script rewrites your train, valid and test set files. We recommend
        you to run this only once.
'''

import os
import sys
import sklearn
import pandas as pd
import numpy as np

data_file = sys.argv[1]
original_tweets = sys.argv[2]

dataset = pd.read_csv(data_file, header=None, sep=',')
dataset = dataset.sample(frac=1)

train_set = dataset.iloc[:int(len(dataset)*0.8), :]
valid_set = dataset.iloc[int(len(dataset)*0.8):int(len(dataset)*0.9), :]
test_set = dataset.iloc[int(len(dataset)*0.9):, :]

train_set.to_csv('train_set.csv')
valid_set.to_csv('train_set.csv')
test_set.to_csv('test_set.csv')

print("New train and test sets saved successfully!")

