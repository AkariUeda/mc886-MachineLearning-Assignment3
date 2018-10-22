import os
import sys
import sklearn
import pandas as pd
import numpy as np

dataset = []

for arq in os.listdir('./dataset'):
    with open('./dataset/'+arq, 'r', errors='ignore') as f:
        data = f.readlines()
    d = []
    for i in range(len(data)):
        d.append(data[i].split('|',2))
    dataset += d
    print(len(dataset))

print(dataset[0])
