#######################
# ----- Imports ----- #
#######################
# NumPy
import numpy

# DefaultDict
from collections import defaultdict

# SkLearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Pandas
import pandas

# matplotlib
import matplotlib.pyplot as pyplot

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

# Utilities
import util

def knnMain(fileName, maxK, minimalOrMutant, showComparisonBetweenNeighbors = False, graphFileName = None):
    if showComparisonBetweenNeighbors == True and graphFileName == None:
        exit()
    
    # Importing CSV
    dataFrame = pandas.read_csv(fileName)

    # Choosing column target
    if minimalOrMutant == 0:
        targetColumn = '_IM_MINIMAL'
    elif minimalOrMutant == 1:
        targetColumn = '_IM_EQUIVALENT'
    else:
        exit()

    # Grouping data frame by target column
    dataGrouped = dataFrame.groupby(targetColumn)
    dataFrame = pandas.DataFrame(dataGrouped.apply(lambda x: x.sample(dataGrouped.size().min()).reset_index(drop=True)))    

    # Get ColumnValues and drop minimal and equivalent columns
    columnValues = dataFrame[targetColumn].values
    dataFrame = dataFrame.drop(['_IM_MINIMAL', '_IM_EQUIVALENT'], axis=1)

    # Encode _IM_OPERATOR column
    one_hot_Operator = pandas.get_dummies(dataFrame['_IM_OPERATOR'])
    dataFrame = dataFrame.drop('_IM_OPERATOR', axis = 1)
    dataFrame = dataFrame.join(one_hot_Operator)

    # Encode _IM_TYPE_STATEMENT column
    one_hot_TypeStatement = pandas.get_dummies(dataFrame['_IM_TYPE_STATEMENT'])
    dataFrame = dataFrame.drop('_IM_TYPE_STATEMENT', axis = 1)
    dataFrame = dataFrame.join(one_hot_TypeStatement)    

    # Get DataFrameValues
    dataFrameValues = dataFrame.values

    # Classifying and calculating scores by each number of neighbors between 1 and k
    accuracy = []
    precision = []
    recall = []
    f1 = []

    data = []

    for kCount in range(1, maxK + 1):
        estimator = KNeighborsClassifier(n_neighbors = kCount)
        
        scores = cross_val_score(estimator, dataFrameValues, columnValues, scoring='accuracy',cv=5)
        accuracy.append(numpy.mean(scores) * 100)
        
        scores = cross_val_score(estimator, dataFrameValues, columnValues, scoring='precision',cv=5)
        precision.append(numpy.mean(scores) * 100)
        
        scores = cross_val_score(estimator, dataFrameValues, columnValues, scoring='recall',cv=5)
        recall.append(numpy.mean(scores) * 100)
        
        scores = cross_val_score(estimator, dataFrameValues, columnValues, scoring='f1',cv=5)
        f1.append(numpy.mean(scores) * 100)
        
        subData = []
        subData.append(kCount)
        subData.append(accuracy[len(accuracy) - 1])
        subData.append(precision[len(precision) - 1])
        subData.append(recall[len(recall) - 1])
        subData.append(f1[len(f1) - 1])

        data.append(subData)
        #print("{:2d} Vizinhos | Acurácia {:.2f}%\tPrecisão: {:.2f}%\tRecall: {:.2f}%\tF1: {:.2f}%".format(kCount, accuracy[len(accuracy) - 1], precision[len(precision) - 1], recall[len(recall) - 1], f1[len(f1) - 1]))

    header = []
    header.append('neighbors')
    header.append('accuracy')
    header.append('precision')
    header.append('recall')
    header.append('f1')
    util.writeInCsvFile('ML/Results/kNN_{targetColumn}.csv'.format(targetColumn = targetColumn), data, header=header)

    if showComparisonBetweenNeighbors:
        plotInfoRateByKValue(maxK, accuracy, precision, recall, f1, graphFileName)

def plotInfoRateByKValue(maxK, accuracy, precision, recall, f1, graphFileName):
    # The next step is to plot the error values against K values. Execute the following script to create the plot:
    pyplot.figure(figsize=(12, 6))
    
    pyplot.plot(range(1, maxK + 1), accuracy,   color='blue',   linestyle='solid', label='Accuracy',    marker='o', markerfacecolor='blue',     markersize=7.5)
    pyplot.plot(range(1, maxK + 1), precision,  color='red',    linestyle='solid', label='Precision',   marker='o', markerfacecolor='red',      markersize=7.5)
    pyplot.plot(range(1, maxK + 1), recall,     color='green',  linestyle='solid', label='Recall',      marker='o', markerfacecolor='green',    markersize=7.5)
    pyplot.plot(range(1, maxK + 1), f1,         color='orange', linestyle='solid', label='F1',          marker='o', markerfacecolor='orange',   markersize=7.5)
    
    pyplot.title('Accuracy Rate K Value')
    pyplot.legend()
    pyplot.xlabel('K Value')
    pyplot.ylabel('Accuracy')
    pyplot.savefig(graphFileName)
    pyplot.show()
    

def computeFullMutants():
    maxK = 40

    print('Calculando para identificar mutantes minimais')
    fileName = 'ML/Mutants/Minimal/With ColumnNames With Operator/1Full Mutants.csv'
    knnMain(fileName, maxK, 0)
    
    print('Calculando para identificar mutantes equivalentes')
    fileName = 'ML/Mutants/Equivalent/With ColumnNames With Operator/1Full Mutants.csv'
    knnMain(fileName, maxK, 1)

def main():
    computeFullMutants()
    #computeMutatsForEachProgram()

if __name__ == '__main__':
    main()