import pandas as pd

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import util

def getDataByFile(file, metric):
    if metric not in ['Accuracy', 'Precision', 'Recall', 'F1']:
        sys.exit()

    data = util.createDataFrameFromCSV(file, True)

    minimal = pd.DataFrame(data).query('TargetColumn == \'MINIMAL\'')
    equivalent = pd.DataFrame(data).query('TargetColumn == \'EQUIVALENT\'')

    return minimal, equivalent

def getData(metric):
    # Data extracted from ML/Results/Summary/Summary_Classifiers
    minimal, equivalent = getDataByFile('{}/ML/Results/Summary/Summary_Classifiers.csv'.format(os.getcwd()), 'F1')

    minimalData = minimal[metric].values
    equivalentData = equivalent[metric].values
    
    return minimalData, equivalentData

def getFullData(metric):
    # Data extracted from ML/Results/Summary/Summary_Classifiers
    minimal, equivalent = getDataByFile('{}/ML/Results/Summary/Summary_BestClassifiers_All30Runs.csv'.format(os.getcwd()), 'F1')

    minimalData = minimal[metric].values
    equivalentData = equivalent[metric].values
    
    return minimalData, equivalentData

# def getFullData(metric):
#     # Data extracted from ML/Results/Summary/Summary_BestClassifiers_All30Runs
#     minimal, equivalent = getDataByFile('{}/ML/Results/Summary/Summary_BestClassifiers_All30Runs.csv'.format(os.getcwd()), 'F1')

#     minimalData = []
#     equivalentData = []

#     for classifier in util.getPossibleClassifiers():
#         minimalMetricFromClassifier = minimal.query('Classifier == \'{}\''.format(classifier))
#         minimalMetricFromClassifier = minimalMetricFromClassifier[metric].tolist()
#         minimalData.append(minimalMetricFromClassifier)

#         equivalentMetricFromClassifier = equivalent.query('Classifier == \'{}\''.format(classifier))
#         equivalentMetricFromClassifier = equivalentMetricFromClassifier[metric].tolist()
#         equivalentData.append(equivalentMetricFromClassifier)

#     return minimalData, equivalentData

if __name__ == '__main__':
    print(getFullData('F1'))