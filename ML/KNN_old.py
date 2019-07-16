# Fonte: https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/

#############################
# --- Importing Libraries ---
#############################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report, confusion_matrix

def importdataSet(fileName, columnNames, showHeadDataSet=False):
    ###############################
    # --- Importing the dataSet ---
    ###############################
        # To import the dataSet and load it into our pandas dataframe, execute the following code:
    #url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    url = fileName

    # --- Read dataSet to pandas dataframe ---
    dataSet = pd.read_csv(url, names=columnNames)

    if showHeadDataSet:
        # --- To see what the dataSet actually looks like, execute the following command: ---
        print(dataSet.head())

    return dataSet

def preProcessing(dataSet, numProperties, testSetSize):
    #######################
    # --- Preprocessing ---
    #######################
        # The next step is to split our dataSet into its attributes and labels. To do so, use the following code:
        # The X variable contains the first four columns of the dataSet (i.e. attributes) while y contains the labels.
    X = dataSet.iloc[:, :-1].values  
    y = dataSet.iloc[:, numProperties].values

    # --- Train Test Split ---
        # To avoid over-fitting, we will divide our dataSet into training and test splits, which gives us a better idea as to how our algorithm performed during the testing phase. This way our algorithm is tested on un-seen data, as it would be in a production application.
        # To create training and test splits, execute the following script:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSetSize)

    # --- Feature Scaling ---
        # Before making any actual predictions, it is always a good practice to scale the features so that all of them can be uniformly evaluated.
        # The gradient descent algorithm (which is used in neural network training and other machine learning algorithms) also converges faster with normalized features.
        # The following script performs feature scaling:
    
    
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def trainingAndPredictions(numberNeighbors, X_train, y_train, X_test):
    ##################################
    # --- Training and Predictions ---
    ##################################
        # It is extremely straight forward to train the KNN algorithm and make predictions with it, especially when using Scikit-Learn.
    classifier = KNeighborsClassifier(n_neighbors=numberNeighbors)
    classifier.fit(X_train, y_train)

        # The final step is to make predictions on our test data. To do so, execute the following script:
    y_pred = classifier.predict(X_test)

    return y_pred

def evaluatingAlgorithm(y_test, y_pred):
    ##################################
    # --- Evaluating the Algorithm ---
    ##################################
        # For evaluating an algorithm, confusion matrix, precision, recall and f1 score are the most commonly used metrics. The confusion_matrix and classification_report methods of the sklearn.metrics can be used to calculate these metrics. Take a look at the following script:
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def comparingErrorRateWithKValue(maxK, X_train, y_train, X_test, y_test):
    ###############################################
    # --- Comparing Error Rate with the K Value ---
    ###############################################
        # In the training and prediction section we said that there is no way to know beforehand which value of K that yields the best results in the first go. We randomly chose 5 as the K value and it just happen to result in 100% accuracy.
        # One way to help you find the best value of K is to plot the graph of K value and the corresponding error rate for the dataSet.
        # In this section, we will plot the mean error for the predicted values of test set for all the K values between 1 and 40.
        # To do so, let's first calculate the mean of error for all the predicted values where K ranges from 1 and 40. Execute the following script:
    error = []

    # Calculating error for K values between 1 and 40
    for i in range(1, maxK):  
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        error.append(np.mean(pred_i != y_test))

    # The next step is to plot the error values against K values. Execute the following script to create the plot:
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, maxK), error, color='red', linestyle='dashed', marker='o',  
            markerfacecolor='blue', markersize=10)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')  
    plt.ylabel('Mean Error')
    plt.show()

def knnMain(fileName, columnNames, numProperties, testSetSize, numberNeighbors, maxK, showComparisonBetweenNeighbors = False):
    ######################
    # ----- Step 1 ----- #
    ######################
    dataSet = importdataSet(fileName, columnNames)
    
    ######################
    # ----- Step 2 ----- #
    ######################
    X_train, X_test, y_train, y_test = preProcessing(dataSet, numProperties, testSetSize)
    
    ######################
    # ----- Step 3 ----- #
    ######################
    y_pred = trainingAndPredictions(numberNeighbors, X_train, y_train, X_test)
    
    ######################
    # ----- Step 4 ----- #
    ######################
    evaluatingAlgorithm(y_test, y_pred)

    if showComparisonBetweenNeighbors:
        ######################
        # ----- Step 5 ----- #
        ######################
        comparingErrorRateWithKValue(maxK, X_train, y_train, X_test, y_test)

def computeMutants():
    fileName = 'ML/Mutants/Minimal/1Full Mutants wOperator.csv'
    #fileName = 'ML/Mutants/Equivalent/1Full Mutants wOperator.csv'
    
    # For minimals
    columnNames = ['_IM_PROGRAM_GRAPH_NODE', '_IM_PRIMITIVE_ARC', '_IM_SOURCE_PRIMITIVE_ARC', '_IM_TARGET_PRIMITIVE_ARC', '_IM_DISTANCE_BEGIN_MIN', '_IM_DISTANCE_BEGIN_MAX', '_IM_DISTANCE_BEGIN_AVG', '_IM_DISTANCE_END_MIN', '_IM_DISTANCE_END_MAX', '_IM_DISTANCE_END_AVG', '_IM_EQUIVALENT', '_IM_MINIMAL']
    # For equivalents
    #columnNames = ['_IM_PROGRAM_GRAPH_NODE', '_IM_PRIMITIVE_ARC', '_IM_SOURCE_PRIMITIVE_ARC', '_IM_TARGET_PRIMITIVE_ARC', '_IM_DISTANCE_BEGIN_MIN', '_IM_DISTANCE_BEGIN_MAX', '_IM_DISTANCE_BEGIN_AVG', '_IM_DISTANCE_END_MIN', '_IM_DISTANCE_END_MAX', '_IM_DISTANCE_END_AVG', '_IM_MINIMAL', '_IM_EQUIVALENT']

    # With operator
    # For minimals
    #columnNames = ['_IM_OPERATOR', '_IM_PROGRAM_GRAPH_NODE', '_IM_PRIMITIVE_ARC', '_IM_SOURCE_PRIMITIVE_ARC', '_IM_TARGET_PRIMITIVE_ARC', '_IM_DISTANCE_BEGIN_MIN', '_IM_DISTANCE_BEGIN_MAX', '_IM_DISTANCE_BEGIN_AVG', '_IM_DISTANCE_END_MIN', '_IM_DISTANCE_END_MAX', '_IM_DISTANCE_END_AVG', '_IM_EQUIVALENT', '_IM_MINIMAL']
    # For equivalents
    #columnNames = ['_IM_OPERATOR', '_IM_PROGRAM_GRAPH_NODE', '_IM_PRIMITIVE_ARC', '_IM_SOURCE_PRIMITIVE_ARC', '_IM_TARGET_PRIMITIVE_ARC', '_IM_DISTANCE_BEGIN_MIN', '_IM_DISTANCE_BEGIN_MAX', '_IM_DISTANCE_BEGIN_AVG', '_IM_DISTANCE_END_MIN', '_IM_DISTANCE_END_MAX', '_IM_DISTANCE_END_AVG', '_IM_MINIMAL', '_IM_EQUIVALENT']    
    
    numProperties = 11 # Without operator
    #numProperties = 12 # With operator

    testSetSize = 0.25

    numberNeighbors = 10

    maxK=40

    knnMain(fileName, columnNames, numProperties, testSetSize, numberNeighbors, maxK)

def computeIris():
    fileName = 'ML/Iris Example/iris.data'
    
    columnNames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
    
    numProperties = 4

    testSetSize = 0.20

    numberNeighbors = 10

    maxK=40

    knnMain(fileName, columnNames, numProperties, testSetSize, numberNeighbors, maxK, True)

if __name__ == '__main__':
    computeMutants()
    #computeIris()