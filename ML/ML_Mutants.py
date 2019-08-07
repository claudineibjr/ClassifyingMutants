# Fonte: https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/

#############################
# --- Importing Libraries ---
#############################
# NumPy
import numpy as np

# MatPlotLib
import matplotlib.pyplot as plt

# Pandas
import pandas as pd

# SkLearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix

# SkLearn - Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Statistics
from statistics import mean
from statistics import median

# Util
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
import util

def importDataSet(fileName, columnNames, showHeadDataSet=False):
    ############################
    # --- Importing the dataSet
    url = fileName

    # --- Read dataSet to pandas dataframe ---
    dataSet = pd.read_csv(url, names=columnNames)

    if showHeadDataSet:
        # --- To see what the dataSet actually looks like, execute the following command: ---
        print(dataSet.head())

    return dataSet

def preProcessing(dataSetFrame, testSetSize, targetColumn, columnsToDrop):
    ####################
    # --- Preprocessing

    dataSetFrame.drop(columnsToDrop, axis = 1)
    numProperties = len(dataSetFrame.columns) - 1

    # Removing duplicates
    #dataSetFrame.drop_duplicates(subset = ['_IM_OPERATOR', '_IM_TYPE_STATEMENT', '_IM_COMPLEXITY'], 
                     #keep = 'first', inplace = True)

    # Grouping data frame by target column
    dataGrouped = dataSetFrame.groupby(targetColumn)
    dataSetFrame = pd.DataFrame(dataGrouped.apply(lambda x: x.sample(dataGrouped.size().min()).reset_index(drop=True)))

    # Número de colunas a serem deletadas
    numColumnsToDelete = 0

    # Encode _IM_OPERATOR column
    if dataSetFrame.columns.__contains__('_IM_OPERATOR'):
        one_hot_Operator = pd.get_dummies(dataSetFrame['_IM_OPERATOR'])
        dataSetFrame = dataSetFrame.drop('_IM_OPERATOR', axis = 1)
        dataSetFrame = dataSetFrame.join(one_hot_Operator)

        numColumnsToDelete = numColumnsToDelete - 1 + len(one_hot_Operator.columns)

    # Encode _IM_TYPE_STATEMENT column
    if dataSetFrame.columns.__contains__('_IM_TYPE_STATEMENT'):
        one_hot_TypeStatement = pd.get_dummies(dataSetFrame['_IM_TYPE_STATEMENT'])
        dataSetFrame = dataSetFrame.drop('_IM_TYPE_STATEMENT', axis = 1)
        dataSetFrame = dataSetFrame.join(one_hot_TypeStatement)
        
        numColumnsToDelete = numColumnsToDelete - 1 + len(one_hot_TypeStatement.columns)

    # Remove a coluna e reinsere-a no final
    targetColumnValues = dataSetFrame[targetColumn]
    dataSetFrame = dataSetFrame.drop(targetColumn, axis = 1)
    dataSetFrame = dataSetFrame.join(targetColumnValues)

    # --- Train Test Split ---
    #   To avoid over-fitting, we will divide our dataSet into training and test splits, which gives us a better idea as to how our algorithm performed during the testing phase. This way our algorithm is tested on un-seen data, as it would be in a production application.
    #   To create training and test splits, execute the following script:
    X = dataSetFrame.iloc[:, :-1].values
    y = dataSetFrame.iloc[:, numProperties + numColumnsToDelete].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSetSize)

    # --- Feature Scaling ---
    #   Before making any actual predictions, it is always a good practice to scale the features so that all of them can be uniformly evaluated.
    #   The gradient descent algorithm (which is used in neural network training and other machine learning algorithms) also converges faster with normalized features.
    #   The following script performs feature scaling:
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def knn_trainingAndPredictions(numberNeighbors, X_train, y_train, X_test):
    ##################################
    # --- Training and Predictions ---
    ##################################
    
    #   It is extremely straight forward to train the KNN algorithm and make predictions with it, especially when using Scikit-Learn.
    classifier = KNeighborsClassifier(n_neighbors=numberNeighbors)
    classifier.fit(X_train, y_train)

    #   The final step is to make predictions on our test data. To do so, execute the following script:
    y_pred = classifier.predict(X_test)

    return y_pred

def dt_trainingAndPredictions(minSamplesSplit, X_train, y_train, X_test):
    ##################################
    # --- Training and Predictions ---
    ##################################
    
    #   It is extremely straight forward to train the KNN algorithm and make predictions with it, especially when using Scikit-Learn.
    classifier = DecisionTreeClassifier(min_samples_split = minSamplesSplit)
    classifier.fit(X_train, y_train)

    #   The final step is to make predictions on our test data. To do so, execute the following script:
    y_pred = classifier.predict(X_test)

    return y_pred

def evaluatingAlgorithm(y_test, y_pred):
    ##################################
    # --- Evaluating the Algorithm ---
    ##################################
    #   For evaluating an algorithm, confusion matrix, precision, recall and f1 score are the most commonly used metrics. The confusion_matrix and classification_report methods of the sklearn.metrics can be used to calculate these metrics. Take a look at the following script:
    confusionMatrix = confusion_matrix(y_test, y_pred)
    TP = confusionMatrix[0][0]  # True Positives
    FN = confusionMatrix[0][1]  # False Negatives
    
    FP = confusionMatrix[1][0]  # False Positives
    TN = confusionMatrix[1][1]  # True Negatives

    ##############
    # --- Métricas
    #   https://medium.com/as-m%C3%A1quinas-que-pensam/m%C3%A9tricas-comuns-em-machine-learning-como-analisar-a-qualidade-de-chat-bots-inteligentes-m%C3%A9tricas-1ba580d7cc96

    error = (FP + FN) / (FP + FN + TP + TN)
    
    ###############
    # --- Accuracy
    #   Provide general information about how many samples are misclassified.
    #    Accuracy is calculated as the sum of correct predictions divided by the total number of predictions
    accuracy = (TP + TN) / (FP + FN + TP + TN)
    
    #########################
    # --- False Positive Rate
    # --- True Positive Rate
    #   The true positive rate (TPR) and false positive rate (FPR) are performance metrics
    #    that are especially useful for imbalanced class problems
    FPR = (FP) / (FP + TN)
    TPR = (TP) / (FN + TP)

    ###############
    # --- Precision
    # --- Recall
    #   Precision (PRE) and recall (REC) are performance metrics that are related to those
    #    true positive and true negative rates, and in fact, recall is synonymous to the true
    #    positive rate.
    precision = (TP) / (TP + FP)
    recall = (TP) / (FN + TP)
    
    ########
    # --- F1
    #   In practice, often a combination of precision and recall is used, the so-called F1-score.
    f1 = 2 * ((precision * recall) / (precision + recall))

    return accuracy, precision, recall, f1, TPR, FPR, TP, FN, FP, TN

def classifierMain(classifier, maxIterations, resultsFileName, X_train, X_test, y_train, y_test, showResults = False):
    # Array com todas as métricas coletadas ao aplicar o algoritmo de ML
    data = []
    arrAccuracy = []
    arrPrecision = []
    arrRecall = []
    arrF1 = []
    arrTPR = []
    arrFPR = []

    arr_y_pred_iter = []

    if classifier == 'KNN':
        for kNeighbors in range(1, maxIterations + 1, 1):
            arr_y_pred_iter.append((knn_trainingAndPredictions(kNeighbors, X_train, y_train, X_test), kNeighbors))
    elif classifier == 'DT':
        for minSamplesSplit in range(5, maxIterations + 1, 10):
            arr_y_pred_iter.append((dt_trainingAndPredictions(minSamplesSplit, X_train, y_train, X_test), minSamplesSplit))
    else:
        return None

    for y_pred, iteration in arr_y_pred_iter:
        accuracy, precision, recall, f1, TPR, FPR, TP, FN, FP, TN = evaluatingAlgorithm(y_test, y_pred)
        accuracy *= 100
        precision *= 100
        recall *= 100
        f1 *= 100
        TPR *= 100
        FPR *= 100

        if showResults:
            print("{:2d} Amostras | Acurácia {:.6f}%\tPrecisão: {:.6f}%\tRecall: {:.6f}%\tF1: {:.6f}%".format(
                iteration, accuracy, precision, recall, f1))

        arrAccuracy.append(accuracy)
        arrPrecision.append(precision)
        arrRecall.append(recall)
        arrF1.append(f1)
        arrTPR.append(TPR)
        arrFPR.append(FPR)

        subData = []
        subData.append(iteration)
        subData.append(accuracy)
        subData.append(precision)
        subData.append(recall)
        subData.append(f1)
        subData.append(TPR)
        subData.append(FPR)
        subData.append(TP)
        subData.append(FN)
        subData.append(FP)
        subData.append(TN)

        data.append(subData)

    header = []
    header.append('SampleSplit')
    header.append('Accuracy')
    header.append('Precision')
    header.append('Recall')
    header.append('F1')
    header.append('TPR')
    header.append('FPR')
    header.append('TP')
    header.append('FN')
    header.append('FP')
    header.append('TN')
    computeData(resultsFileName, header, data, arrAccuracy, arrPrecision, arrRecall, arrF1)


def computeData(resultsFileName, header, data, accuracy, precision, recall, f1):
    # Minímo
    subData = []
    subData.append('Min')
    subData.append(min(accuracy))   # Accuracy
    subData.append(min(precision))  # Precision
    subData.append(min(recall))     # Recall
    subData.append(min(f1))         # F1
    data.append('')
    data.append(subData)

    # Máximo
    subData = []
    subData.append('Max')
    subData.append(max(accuracy))   # Accuracy
    subData.append(max(precision))  # Precision
    subData.append(max(recall))     # Recall
    subData.append(max(f1))         # F1
    data.append(subData)

    # Average
    subData = []
    subData.append('Mean')
    subData.append(mean(accuracy))   # Accuracy
    subData.append(mean(precision))  # Precision
    subData.append(mean(recall))     # Recall
    subData.append(mean(f1))         # F1
    data.append(subData)

    # Median
    subData = []
    subData.append('Median')
    subData.append(median(accuracy))   # Accuracy
    subData.append(median(precision))  # Precision
    subData.append(median(recall))     # Recall
    subData.append(median(f1))         # F1
    data.append(subData)

    # Print
    util.writeInCsvFile(resultsFileName, data, header=header)    

def computeMutants(columnsToDrop = [], printResults = False):
    ######################
    # --- Setting datasets
    fileNameMinimalDataSet = 'ML/Mutants/Minimal/Without ColumnNames/mutants_minimals.csv'
    fileNameEquivalentsDataSet = 'ML/Mutants/Equivalent/Without ColumnNames/mutants_equivalents.csv'
    
    #####################
    # --- Setting columns
    # For minimals
    minimalColumnNames = ['_IM_OPERATOR', '_IM_SOURCE_PRIMITIVE_ARC', '_IM_TARGET_PRIMITIVE_ARC', '_IM_DISTANCE_BEGIN_MIN', '_IM_DISTANCE_BEGIN_MAX', '_IM_DISTANCE_BEGIN_AVG', '_IM_DISTANCE_END_MIN', '_IM_DISTANCE_END_MAX', '_IM_DISTANCE_END_AVG', '_IM_COMPLEXITY', '_IM_TYPE_STATEMENT', '_IM_EQUIVALENT', '_IM_MINIMAL']
    
    # For equivalents
    equivalentColumnNames = ['_IM_OPERATOR', '_IM_SOURCE_PRIMITIVE_ARC', '_IM_TARGET_PRIMITIVE_ARC', '_IM_DISTANCE_BEGIN_MIN', '_IM_DISTANCE_BEGIN_MAX', '_IM_DISTANCE_BEGIN_AVG', '_IM_DISTANCE_END_MIN', '_IM_DISTANCE_END_MAX', '_IM_DISTANCE_END_AVG', '_IM_COMPLEXITY', '_IM_TYPE_STATEMENT', '_IM_MINIMAL', '_IM_EQUIVALENT']
    
    ###############################
    # --- Setting others properties
    testSetSize = 0.25

    maxNeighbors = 40
    maxSamplesSplit = 100

    ###################
    # --- PreProcessing
    #   Minimals
    targetColumn = '_IM_MINIMAL'
    minimalDataSet = importDataSet(fileNameMinimalDataSet, minimalColumnNames)
    X_train_minimal, X_test_minimal, y_train_minimal, y_test_minimal = preProcessing(minimalDataSet, testSetSize, targetColumn, columnsToDrop)
    
    #   Equivalents
    targetColumn = '_IM_EQUIVALENT'
    equivalentDataSet = importDataSet(fileNameEquivalentsDataSet, equivalentColumnNames)
    X_train_equivalents, X_test_equivalents, y_train_equivalents, y_test_equivalents = preProcessing(equivalentDataSet, testSetSize, targetColumn, columnsToDrop)
    
    #####################################
    # --- Executing kNN and Decision Tree
    #   Minimals
    print('##########################################################')
    print(' ----- KNN - Calculando para identificar mutantes minimais')

    targetColumn = '_IM_MINIMAL'
    resultsFileName = 'ML/Results/kNN_{targetColumn}.csv'.format(targetColumn = targetColumn)
    if len(columnsToDrop) > 0:
        resultsFileName = 'ML/Results/new/kNN_{targetColumn} - {columns}.csv'.format(targetColumn = targetColumn, columns=columnsToDrop)
    classifierMain('KNN', maxNeighbors, resultsFileName, X_train_minimal, X_test_minimal, y_train_minimal, y_test_minimal, printResults)
    
    print(' ------ DT - Calculando para identificar mutantes minimais')

    resultsFileName = 'ML/Results/DT_{targetColumn}.csv'.format(targetColumn = targetColumn)
    if len(columnsToDrop) > 0:
        resultsFileName = 'ML/Results/new/DT_{targetColumn} - {columns}.csv'.format(targetColumn = targetColumn, columns=columnsToDrop)
    classifierMain('DT', maxSamplesSplit, resultsFileName, X_train_minimal, X_test_minimal, y_train_minimal, y_test_minimal, printResults)

    #   Equivalents
    print('\n')
    print('##############################################################')
    print(' ----- KNN - Calculando para identificar mutantes equivalentes')

    targetColumn = '_IM_EQUIVALENT'
    resultsFileName = 'ML/Results/kNN_{targetColumn}.csv'.format(targetColumn = targetColumn)
    if len(columnsToDrop) > 0:
        resultsFileName = 'ML/Results/new/kNN_{targetColumn} - {columns}.csv'.format(targetColumn = targetColumn, columns=columnsToDrop)
    classifierMain('KNN', maxNeighbors, resultsFileName, X_train_equivalents, X_test_equivalents, y_train_equivalents, y_test_equivalents, printResults)
    
    print(' ------ DT - Calculando para identificar mutantes equivalentes')
    
    resultsFileName = 'ML/Results/DT_{targetColumn}.csv'.format(targetColumn = targetColumn)
    if len(columnsToDrop) > 0:
        resultsFileName = 'ML/Results/new/DT_{targetColumn} - {columns}.csv'.format(targetColumn = targetColumn, columns=columnsToDrop)    
    classifierMain('DT', maxSamplesSplit, resultsFileName, X_train_equivalents, X_test_equivalents, y_train_equivalents, y_test_equivalents, printResults)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        computeMutants(printResults = sys.argv[1])
    else:
        computeMutants()