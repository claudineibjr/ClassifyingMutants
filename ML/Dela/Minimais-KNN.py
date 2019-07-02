#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'ML/Dela'))
	print(os.getcwd())
except:
	pass

#%%
# Imports
from sklearn.datasets import load_iris
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.neighbors import KNeighborsClassifier

import pandas as pd

#%% Importing CSV
dataFrame = pd.read_csv('1Full Mutants wOperator.csv')

#escolher um desses dois
column = '_IM_MINIMAL'
#column = '_IM_EQUIVALENT'

g = dataFrame.groupby(column)
dataFrame = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True)))
print(dataFrame[column].value_counts())

columnValues = dataFrame[column].values
#dataFrame.drop(['_IM_MINIMAL','_IM_EQUIVALENT'], axis=1)
dataFrame.drop(['_IM_MINIMAL'], axis=1)
dataFrameValues = dataFrame.values


#%%
# Calculating scores (accuracy, precision, recall and f1)
maxK = 40

accuracy = []
precision = []
recall = []
f1 = []

for kCount in range(1, maxK + 1):
    estimator = KNeighborsClassifier(n_neighbors=kCount)
    
    scores = cross_val_score(estimator, dataFrameValues, columnValues, scoring='accuracy',cv=5)
    accuracy.append(np.mean(scores) * 100)
    
    scores = cross_val_score(estimator, dataFrameValues, columnValues, scoring='precision',cv=5)
    precision.append(np.mean(scores) * 100)
    
    scores = cross_val_score(estimator, dataFrameValues, columnValues, scoring='recall',cv=5)
    recall.append(np.mean(scores) * 100)
    
    scores = cross_val_score(estimator, dataFrameValues, columnValues, scoring='f1',cv=5)
    f1.append(np.mean(scores) * 100)
    
    print("{:2d} Vizinhos Acurácia | {:.2f}% Precisão: {:.2f}% Recall: {:.2f}% F1: {:.2f}%".format(kCount, accuracy[len(accuracy) - 1], precision[len(precision) - 1], recall[len(recall) - 1], f1[len(f1) - 1]))