# Fonte: https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/

#############################
# --- Importing Libraries ---
#############################

# ---------
# --- NumPy
import numpy as np

# --------------
# --- MatPlotLib
import matplotlib.pyplot as plt

# ----------
# --- Pandas
import pandas as pd

# -----------
# --- SkLearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import VarianceThreshold

# -------------------------
# --- SkLearn - Classifiers
#https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
# KNN - K Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

#https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
# DT - Decision Tree
from sklearn.tree import DecisionTreeClassifier

#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# RF - Random Forest
from sklearn.ensemble import RandomForestClassifier

#https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# SVM - Support Vector Machine
from sklearn.svm import SVC

#https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html
# LDA - Linear Discriminant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
# LR - Logistic Regression
from sklearn.linear_model import LogisticRegression

#https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
# GNB - Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB

# --------------
# --- Statistics
from statistics import mean
from statistics import median

# ------------
# --- Analyzes
#from analyzes import getBestClassifierForPrograms
import analyzes

# Ignoring FutureWarning
import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)
#import warnings
warnings.filterwarnings("ignore")

# --------
# --- Util
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import util

from util import getPossibleClassifiers, getFullNamePossibleClassifiers, getPossibleTargetColumns

def getColumnNames():
	return ['_IM_PROGRAM', '_IM_OPERATOR', '_IM_SOURCE_PRIMITIVE_ARC', '_IM_TARGET_PRIMITIVE_ARC', '_IM_DISTANCE_BEGIN_MIN', '_IM_DISTANCE_BEGIN_MAX', '_IM_DISTANCE_BEGIN_AVG', '_IM_DISTANCE_END_MIN', '_IM_DISTANCE_END_MAX', '_IM_DISTANCE_END_AVG', '_IM_COMPLEXITY', '_IM_TYPE_STATEMENT', '_IM_MINIMAL', '_IM_EQUIVALENT']

def getColumnNames_lastMinimal():
	return ['_IM_PROGRAM', '_IM_OPERATOR', '_IM_SOURCE_PRIMITIVE_ARC', '_IM_TARGET_PRIMITIVE_ARC', '_IM_DISTANCE_BEGIN_MIN', '_IM_DISTANCE_BEGIN_MAX', '_IM_DISTANCE_BEGIN_AVG', '_IM_DISTANCE_END_MIN', '_IM_DISTANCE_END_MAX', '_IM_DISTANCE_END_AVG', '_IM_COMPLEXITY', '_IM_TYPE_STATEMENT', '_IM_EQUIVALENT', '_IM_MINIMAL']

def getColumnNames_lastEquivalent():
	return ['_IM_PROGRAM', '_IM_OPERATOR', '_IM_SOURCE_PRIMITIVE_ARC', '_IM_TARGET_PRIMITIVE_ARC', '_IM_DISTANCE_BEGIN_MIN', '_IM_DISTANCE_BEGIN_MAX', '_IM_DISTANCE_BEGIN_AVG', '_IM_DISTANCE_END_MIN', '_IM_DISTANCE_END_MAX', '_IM_DISTANCE_END_AVG', '_IM_COMPLEXITY', '_IM_TYPE_STATEMENT', '_IM_MINIMAL', '_IM_EQUIVALENT']

def getPossiblePrograms():
	possiblePrograms = [util.getPathName(program) for program in util.getPrograms('{}/Programs'.format(os.getcwd()))]
	return possiblePrograms

def importDataSet(fileName, columnNames, showHeadDataSet=False):
	############################
	# --- Importing the dataSet
	url = fileName

	# --- Read dataSet to pandas dataframe
	dataSet = pd.read_csv(url, names=columnNames, header=0)

	if showHeadDataSet:
		# --- To see what the dataSet actually looks like, execute the following command
		print(dataSet.head())

	return dataSet

def preProcessing(dataSetFrame, targetColumn, columnNames, columnsToDrop, columnsToAdd, allOperators = None, allTypeStatement = None, groupByTargetColumn = True):
	####################
	# --- Preprocessing

	# Add or remove due columns
	if len(columnsToDrop) > 0:
		dataSetFrame = dataSetFrame.drop(columnsToDrop, axis = 1)
	elif len(columnsToAdd) > 0:
		for column in columnNames:
			if column not in columnsToAdd and len(column) > 1 and column != '_IM_MINIMAL' and column != '_IM_EQUIVALENT':
				dataSetFrame = dataSetFrame.drop(column, axis = 1)

	numProperties = len(dataSetFrame.columns) - 1

	#Preprocessing columns that will be 

	# Grouping data frame by target column
	if groupByTargetColumn:
		dataGrouped = dataSetFrame.groupby(targetColumn)
		dataSetFrame = pd.DataFrame(dataGrouped.apply(lambda x: x.sample(dataGrouped.size().min()).reset_index(drop = True)))

	groupedDataSetFrame = dataSetFrame.copy()

	# Remove the program name
	if dataSetFrame.columns.__contains__('_IM_PROGRAM'):
		dataSetFrame = dataSetFrame.drop('_IM_PROGRAM', axis = 1)
		numProperties -= 1

	# Columns number to be deleted
	numColumnsToDelete = 0

	# Encode _IM_OPERATOR column
	if dataSetFrame.columns.__contains__('_IM_OPERATOR'):
		allPossibleOperators = dataSetFrame['_IM_OPERATOR'].values
		one_hot_Operator = pd.get_dummies(dataSetFrame['_IM_OPERATOR'])

		if allOperators is not None:
			operatorsNotInDataSet = list(set(allOperators) - set(allPossibleOperators))
			for operator in operatorsNotInDataSet:
				one_hot_Operator.insert(len(one_hot_Operator.columns) - 1, operator, 0)
			
		dataSetFrame = dataSetFrame.drop('_IM_OPERATOR', axis = 1)
		dataSetFrame = dataSetFrame.join(one_hot_Operator)

		numColumnsToDelete = numColumnsToDelete - 1 + len(one_hot_Operator.columns)

	# Encode _IM_TYPE_STATEMENT column
	if dataSetFrame.columns.__contains__('_IM_TYPE_STATEMENT'):
		allPossibleTypeStatement = dataSetFrame['_IM_TYPE_STATEMENT'].values
		one_hot_TypeStatement = pd.get_dummies(dataSetFrame['_IM_TYPE_STATEMENT'])

		if allTypeStatement is not None:
			typeStatementNotInDataSet = list(set(allTypeStatement) - set(allPossibleTypeStatement))
			for typeStatement in typeStatementNotInDataSet:
				one_hot_TypeStatement.insert(len(one_hot_TypeStatement.columns) - 1, typeStatement, 0)

		dataSetFrame = dataSetFrame.drop('_IM_TYPE_STATEMENT', axis = 1)
		dataSetFrame = dataSetFrame.join(one_hot_TypeStatement)

		numColumnsToDelete = numColumnsToDelete - 1 + len(one_hot_TypeStatement.columns)

	# Remove the target column and reinsert it at final
	targetColumnValues = dataSetFrame[targetColumn]
	dataSetFrame = dataSetFrame.drop(targetColumn, axis = 1)
	dataSetFrame = dataSetFrame.join(targetColumnValues)

	return dataSetFrame, numProperties, numColumnsToDelete, allPossibleOperators, allPossibleTypeStatement, groupedDataSetFrame

def dataSplittingIntoTrainAndTest(dataSetFrame, numProperties, numColumnsToDelete, testSetSize):
	# --- Train Test Split ---
	#   To avoid over-fitting, we will divide our dataSet into training and test splits, which gives us a better idea as to how our algorithm performed during the testing phase. This way our algorithm is tested on un-seen data, as it would be in a production application.
	#   To create training and test splits, execute the following script:
	X = dataSetFrame.iloc[:, :-1].values
	y = dataSetFrame.iloc[:, numProperties + numColumnsToDelete].values
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSetSize)

	# --- Feature Scaling ---
	#   Before making any actual predictions, it is always a good practice to scale the features so that all of them can be uniformly evaluated.
	#   The gradient descent algorithm (which is used in neural network training and other machine learning algorithms) also converges faster with normalized features.
	#   The following script performs feature scaling:
	scaler = StandardScaler()
	scaler.fit(X_train)
	
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)

	return X_train, X_test, y_train, y_test

'''
	Parameter could be the numNeighbors for kNN and minSampleSplit for DecisionTree and RandomForest
'''
def trainingAndPredictions(strClassifier, parameter, X_train, y_train, X_test):
	##################################
	# --- Training and Predictions ---
	##################################
	
	#   It is extremely straight forward to train the KNN algorithm and make predictions with it, especially when using Scikit-Learn.
	if strClassifier.upper() == 'KNN':
		classifier = KNeighborsClassifier(n_neighbors = parameter)
	elif strClassifier.upper() == 'DT':
		classifier = DecisionTreeClassifier(min_samples_split = parameter)
	elif strClassifier.upper() == 'RF':
		classifier = RandomForestClassifier(min_samples_split = parameter)
	elif strClassifier.upper() == 'SVM':
		classifier = SVC()
	elif strClassifier.upper() == 'LDA':
		classifier = LinearDiscriminantAnalysis()
	elif strClassifier.upper() == 'LR':
		classifier = LogisticRegression()
	elif strClassifier.upper() == 'GNB':
		classifier = GaussianNB()
	else:
		return None

	classifier.fit(X_train, y_train)
	
	#   The final step is to make predictions on our test data. To do so, execute the following script:
	y_pred = classifier.predict(X_test)

	return y_pred

def evaluatingClassification(y_test, y_pred):
	##################################
	# --- Evaluating the Algorithm ---
	##################################
	#   For evaluating an algorithm, confusion matrix, precision, recall and f1 score are the most commonly used metrics. The confusion_matrix and classification_report methods of the sklearn.metrics can be used to calculate these metrics. Take a look at the following script:
	confusionMatrix = confusion_matrix(y_test, y_pred)
	TP = confusionMatrix[1][1]  # True Positives
	FN = confusionMatrix[1][0]  # False Negatives
	
	FP = confusionMatrix[0][1]  # False Positives
	TN = confusionMatrix[0][0]  # True Negatives

	#print(confusionMatrix)
	#print('TP: {} | FN: {} | FP: {} | TN: {}'.format(TP, FN, FP, TN))

	##############
	# --- Metrics
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

def classifierMain(classifier, maxIterations, resultsFileName, X_train, X_test, y_train, y_test, showResults = False, parameter = None):
	# Arrays containing all collected metrics on applying Machine Learning algorithm
	data = []
	arrAccuracy = []
	arrPrecision = []
	arrRecall = []
	arrF1 = []
	arrTPR = []
	arrFPR = []

	arr_y_pred_iter = []

	if classifier == 'KNN':
		if parameter is None:
			for kNeighbors in range(1, maxIterations + 1, 1):
				arr_y_pred_iter.append((trainingAndPredictions('KNN', kNeighbors, X_train, y_train, X_test), kNeighbors))
		else:
			arr_y_pred_iter.append((trainingAndPredictions('KNN', kNeighbors, X_train, y_train, X_test), parameter))
	elif classifier == 'DT':
		if parameter is None:
			for minSamplesSplit in range(5, maxIterations + 1, 10):
				arr_y_pred_iter.append((trainingAndPredictions('DT', minSamplesSplit, X_train, y_train, X_test), minSamplesSplit))
		else:
			arr_y_pred_iter.append((trainingAndPredictions('DT', minSamplesSplit, X_train, y_train, X_test), parameter))
	elif classifier == 'RF':
		if parameter is None:
			for minSamplesSplit in range(5, maxIterations + 1, 10):
				arr_y_pred_iter.append((trainingAndPredictions('RF', minSamplesSplit, X_train, y_train, X_test), minSamplesSplit))
		else:
			arr_y_pred_iter.append((trainingAndPredictions('RF', minSamplesSplit, X_train, y_train, X_test), parameter))
	elif classifier == 'SVM': #TODO
		if parameter is None:
			for minSamplesSplit in range(5, maxIterations + 1, 10):
				arr_y_pred_iter.append((trainingAndPredictions('SVM', minSamplesSplit, X_train, y_train, X_test), minSamplesSplit))
		else:
			arr_y_pred_iter.append((trainingAndPredictions('SVM', minSamplesSplit, X_train, y_train, X_test), parameter))			
	else:
		return None

	for y_pred, iteration in arr_y_pred_iter:
		accuracy, precision, recall, f1, TPR, FPR, TP, FN, FP, TN = evaluatingClassification(y_test, y_pred)
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

def crossValidation_main(dataSetFrame, targetColumn, classifier, maxIterations, resultsFileName, columnNames, columnsToDrop, columnsToAdd, showResults = False, parameter = None):

	# Add or remove due columns
	if len(columnsToDrop) > 0:
		dataSetFrame.drop(columnsToDrop, axis = 1)
	elif len(columnsToAdd) > 0:
		for column in columnNames:
			if column not in columnsToAdd and len(column) > 1 and column != '_IM_MINIMAL' and column != '_IM_EQUIVALENT':
				dataSetFrame = dataSetFrame.drop(column, axis = 1)

	# Grouping data frame by target column
	dataGrouped = dataSetFrame.groupby(targetColumn)
	dataSetFrame = pd.DataFrame(dataGrouped.apply(lambda x: x.sample(dataGrouped.size().min()).reset_index(drop = True)))

	# Remove the program name
	if dataSetFrame.columns.__contains__('_IM_PROGRAM'):
		dataSetFrame = dataSetFrame.drop('_IM_PROGRAM', axis = 1)

	# Get ColumnValues and drop minimal and equivalent columns
	columnValues = dataSetFrame[targetColumn].values
	dataSetFrame = dataSetFrame.drop(['_IM_MINIMAL', '_IM_EQUIVALENT'], axis = 1)

	# Encode _IM_OPERATOR column
	if dataSetFrame.columns.__contains__('_IM_OPERATOR'):
		one_hot_Operator = pd.get_dummies(dataSetFrame['_IM_OPERATOR'])
		dataSetFrame = dataSetFrame.drop('_IM_OPERATOR', axis = 1)
		dataSetFrame = dataSetFrame.join(one_hot_Operator)

	# Encode _IM_TYPE_STATEMENT column
	if dataSetFrame.columns.__contains__('_IM_TYPE_STATEMENT'):
		one_hot_TypeStatement = pd.get_dummies(dataSetFrame['_IM_TYPE_STATEMENT'])
		dataSetFrame = dataSetFrame.drop('_IM_TYPE_STATEMENT', axis = 1)
		dataSetFrame = dataSetFrame.join(one_hot_TypeStatement)

	# Get DataFrameValues
	dataFrameValues = dataSetFrame.values

	# Arrays containing all collected metrics on applying Machine Learning algorithm
	data = []
	arrAccuracy = []
	arrPrecision = []
	arrRecall = []
	arrF1 = []

	arr_estimators_iter = []

	if classifier == 'KNN':
		if parameter is None:
			# If the row number are lower the iterations number, it is necessary to iterate just 75% the row number
			maxIterations = maxIterations if len(columnValues) > maxIterations else int(len(columnValues) * 0.75)
			for kNeighbors in range(1, maxIterations + 1, 1):
				arr_estimators_iter.append((KNeighborsClassifier(n_neighbors = kNeighbors), kNeighbors))
		else:
			arr_estimators_iter.append((KNeighborsClassifier(n_neighbors = parameter), parameter))
	elif classifier == 'DT':
		if parameter is None:
			for minSamplesSplit in range(5, maxIterations + 1, 10):
				arr_estimators_iter.append((DecisionTreeClassifier(min_samples_split = minSamplesSplit), minSamplesSplit))
		else:
			arr_estimators_iter.append((DecisionTreeClassifier(min_samples_split = parameter), parameter))
	elif classifier == 'RF':
		if parameter is None:
			for minSamplesSplit in range(5, maxIterations + 1, 10):
				arr_estimators_iter.append((RandomForestClassifier(min_samples_split = minSamplesSplit), minSamplesSplit))
		else:
			arr_estimators_iter.append((RandomForestClassifier(min_samples_split = parameter), parameter))
	elif classifier == 'SVM':
		arr_estimators_iter.append((SVC(), 0))
	elif classifier == 'LDA':
		arr_estimators_iter.append((LinearDiscriminantAnalysis(), 0))
	elif classifier == 'LR':
		arr_estimators_iter.append((LogisticRegression(), 0))
	elif classifier == 'GNB':
		arr_estimators_iter.append((GaussianNB(), 0))
	else:
		return None

	for classifier, iteration in arr_estimators_iter:
		# If the row number of each class are lower than 10, it is necessary set this value to KFold
		n_splits = 10 if len(columnValues) / 2 > 10 else int(len(columnValues) / 2) # Number of folds in a `(Stratified)KFold - :term:` CV splitter

		scores = cross_val_score(classifier, dataFrameValues, columnValues, scoring='accuracy', cv = n_splits)
		arrAccuracy.append(np.mean(scores) * 100)
		
		scores = cross_val_score(classifier, dataFrameValues, columnValues, scoring='precision', cv = n_splits)
		arrPrecision.append(np.mean(scores) * 100)
		
		scores = cross_val_score(classifier, dataFrameValues, columnValues, scoring='recall', cv = n_splits)
		arrRecall.append(np.mean(scores) * 100)
		
		scores = cross_val_score(classifier, dataFrameValues, columnValues, scoring='f1', cv = n_splits)
		arrF1.append(np.mean(scores) * 100)
		
		if showResults:
			print("{:2d} Amostras | Acurácia {:.6f}%\tPrecisão: {:.6f}%\tRecall: {:.6f}%\tF1: {:.6f}%".format(
				iteration, arrAccuracy[len(arrAccuracy) - 1], arrPrecision[len(arrPrecision) - 1], arrRecall[len(arrRecall) - 1], arrF1[len(arrF1) - 1]))

		subData = []
		subData.append(iteration)
		subData.append(arrAccuracy[len(arrAccuracy) - 1])
		subData.append(arrPrecision[len(arrPrecision) - 1])
		subData.append(arrRecall[len(arrRecall) - 1])
		subData.append(arrF1[len(arrF1) - 1])

		data.append(subData)

	header = []
	header.append('SampleSplit')
	header.append('Accuracy')
	header.append('Precision')
	header.append('Recall')
	header.append('F1')
	computeData(resultsFileName, header, data, arrAccuracy, arrPrecision, arrRecall, arrF1)

def computeData(resultsFileName, header, data, accuracy, precision, recall, f1):
	newData = []
	
	# Minimum
	subData = []
	subData.append('Min')
	subData.append(min(accuracy))   # Accuracy
	subData.append(min(precision))  # Precision
	subData.append(min(recall))     # Recall
	subData.append(min(f1))         # F1
	newData.append(subData)

	# Maximum
	subData = []
	subData.append('Max')
	subData.append(max(accuracy))   # Accuracy
	subData.append(max(precision))  # Precision
	subData.append(max(recall))     # Recall
	subData.append(max(f1))         # F1
	newData.append(subData)

	# Average
	subData = []
	subData.append('Mean')
	subData.append(mean(accuracy))   # Accuracy
	subData.append(mean(precision))  # Precision
	subData.append(mean(recall))     # Recall
	subData.append(mean(f1))         # F1
	newData.append(subData)

	# Median
	subData = []
	subData.append('Median')
	subData.append(median(accuracy))   # Accuracy
	subData.append(median(precision))  # Precision
	subData.append(median(recall))     # Recall
	subData.append(median(f1))         # F1
	newData.append(subData)

	# Include each data in the file
	newData.append('')
	newData.append(header)
	for _data in data:
		newData.append(_data)

	# Print
	util.writeInCsvFile(resultsFileName, newData)

def crossValidation(targetColumn, classifier, specifiedProgram = None, columnsToDrop = [], columnsToAdd = [], printResults = False, parameter = None):
	if not classifier in getPossibleClassifiers() or not targetColumn in getPossibleTargetColumns():
		return None
	
	####################################
	# --- Setting independent properties
	maxNeighbors = 40
	maxSamplesSplit = 100
	maxIterations = maxNeighbors if classifier == 'KNN' else maxSamplesSplit

	######################
	# --- Setting datasets
	targetColumnName = targetColumn
	targetColumn = '_IM_{}'.format(targetColumn)
	
	# Verify if it setted a specified program to be classified
	if not specifiedProgram is None:
		dataSetFileName = 'ML/Dataset/{}/Programs/{}.csv'.format(targetColumnName, specifiedProgram)
	else:
		dataSetFileName = 'ML/Dataset/{}/mutants.csv'.format(targetColumnName)

	if targetColumn == '_IM_MINIMAL':
		#####################
		# --- Setting columns
		columnNames = getColumnNames_lastMinimal()

		print('####################################################')
		print(' ----- Calculando para identificar mutantes minimais')
	
	elif targetColumn == '_IM_EQUIVALENT':
		#####################
		# --- Setting columns
		columnNames = getColumnNames_lastEquivalent()

		print('########################################################')
		print(' ----- Calculando para identificar mutantes equivalentes')
	else:
		return

	###################
	# --- PreProcessing
	dataSet = importDataSet(dataSetFileName, columnNames)

	##############################
	# --- Setting results filename
	bestParameter = '_bestParameter' if not parameter is None else ''
	gbs = ' - gbs_{columns}' if len(columnsToDrop) > 0 else ''
	gfs = ' - gfs_{columns}' if len(columnsToAdd) > 0 else ''
	if specifiedProgram is None:
		resultsFileName = 'ML/Results/{targetColumnName}/{classifier}{bestParameter}{gbs}{gfs}.csv'.format(targetColumnName = targetColumnName, classifier = classifier, gbs = gbs, gfs = gfs, bestParameter = bestParameter)
	else:
		resultsFileName = 'ML/Results/{targetColumnName}/Programs/{specifiedProgram}_{classifier}{bestParameter}.csv'.format(targetColumnName = targetColumnName, specifiedProgram = specifiedProgram, classifier = classifier, bestParameter = bestParameter)

	##########################################
	# --- Executing classifier | KNN, DT ou RF
	print(' ----- {}'.format(classifier))
	crossValidation_main(dataSet, targetColumn, classifier, maxIterations, resultsFileName, columnNames, columnsToDrop, columnsToAdd, parameter=parameter)

def classify(newDataSetFileName, resultDataSetFileName, targetColumn, classifier, algorithmParameter, programToClassify):
	""" Function responsable to classify a new data set as equivalent or minimal from predictive models are existing

		Args:
			newDataSetFileName (str): File name containing new mutants to be classified
			resultDataSetFileName (str): File name to be generated with the classification result. This file contains the same row number than 'newDataSetFileName'.
			targetColumn (str): Column to be classified. Must be 'MINIMAL' ou 'EQUIVALENT'.
			classifier (str): The classifier algorithm used to predict the new data inputed. Must be 'KNN', 'DT', 'RF', 'SVM', 'LDA', 'LR' or 'GNB'
			algorithmParameter (int): The parameter to be used on classifier. This parameter Must be K, as the number of neighbors on KNN, or min sample split to Decision Tree and RandomForest.
			programToClassify(str): ---
	"""

	######################
	# --- Setting datasets
	targetColumnName = targetColumn
	targetColumn = '_IM_{}'.format(targetColumn)
	trainDataSetFileName = 'ML/Dataset/{}/mutants.csv'.format(targetColumnName)

	if targetColumn == '_IM_MINIMAL':
		#####################
		# --- Setting columns
		columnNames = getColumnNames_lastMinimal()
	
	elif targetColumn == '_IM_EQUIVALENT':
		#####################
		# --- Setting columns
		 columnNames = getColumnNames_lastEquivalent()

	###################
	# --- PreProcessing

	# --- Import
	trainDataSet = importDataSet(trainDataSetFileName, columnNames)
	trainDataSet = trainDataSet.query('_IM_PROGRAM != \'{}\''.format(programToClassify))
	newDataSetFrame = importDataSet(newDataSetFileName, columnNames)

	# --- PreProccess
	operatorsToTrain = list(set(trainDataSet['_IM_OPERATOR'].values))
	typeStatementsToTrain = list(set(trainDataSet['_IM_TYPE_STATEMENT'].values))
	operatorsToTest = list(set(newDataSetFrame['_IM_OPERATOR'].values))
	typeStatementsToTest = list(set(newDataSetFrame['_IM_TYPE_STATEMENT'].values))

	allOperators = list(set(operatorsToTrain + operatorsToTest))
	allTypeStatement = list(set(typeStatementsToTrain + typeStatementsToTest))

	trainDataSetFrame, numProperties, numColumnsToDelete_train, _, _, groupedDataSetFrame = preProcessing(trainDataSet, targetColumn, columnNames, [], [], allOperators, allTypeStatement)
	newDataSetFrame, numProperties, numColumnsToDelete_test, _, _, groupedDataSetFrame = preProcessing(newDataSetFrame, targetColumn, columnNames, [], [], allOperators, allTypeStatement, False)

	# Separate the data into X (values) and y (target value)
	X_train = trainDataSetFrame.iloc[:, :-1].values
	X_test = newDataSetFrame.iloc[:, :-1].values
	y_train = trainDataSetFrame.iloc[:, numProperties + numColumnsToDelete_train].values

	##############################################################################
	# --- Classify and write new CSV with informations about the prediction result
	y_test = trainingAndPredictions(classifier, algorithmParameter, X_train, y_train, X_test)
	
	# Create an array with the results of prediction | 1 for correct, 0 for incorrect
	result = [1 if predicted == groupedDataSetFrame[targetColumn][iCount] else 0 for iCount, predicted in zip(range(len(y_test)), y_test)]
	
	##############################
	# --- Metrics about prediction
	#total = len(result)
	#correct = result.count(1)
	#perc = correct * 100 / total
	#print('Total: {} | Correto: {} | Perc: {}'.format(total, correct, perc))

	predictedDF = pd.DataFrame(groupedDataSetFrame)
	predictedDF['PREDICTED'] = y_test
	predictedDF['RESULT'] = result

	onlyResultDataSetFileName = str(resultDataSetFileName).replace('.csv', '_result.csv')
	util.writeInCsvFile(onlyResultDataSetFileName, [str(value) for value in y_test])
	util.writeDataFrameInCsvFile(resultDataSetFileName, predictedDF)

def executeAll(targetColumns, classifiers, specifiedProgram = None, executeWithBestParameter = False):
	'''
		Function used to execute all classifiers in all columns to be sorted
	'''
	for column in targetColumns:
		for classifier in getPossibleClassifiers():
			parameter = bestParameter(column, classifier) if executeWithBestParameter else None
			crossValidation(column, classifier, specifiedProgram, parameter=parameter)

def executeAllEachProgram(targetColumns, classifiers, programs, executeWithBestParameter = False):
	for program in programs:
		executeAll(targetColumns, classifiers, program, executeWithBestParameter)

def bestParameter(targetColumn, classifier):
	key = '{}_{}'.format(targetColumn, classifier) #Column_Classifier
	
	parameters = dict()
	parameters['MINIMAL_KNN'] = 5
	parameters['MINIMAL_DT'] = 15
	parameters['MINIMAL_RF'] = 5
	parameters['EQUIVALENT_KNN'] = 3
	parameters['EQUIVALENT_DT'] = 15
	parameters['EQUIVALENT_RF'] = 5

	if key in parameters.keys():
		return parameters[key]
	else:
		return None

def debug_main(arguments):
	'''
		Main function performed at the time of running the experiment
	'''
	# Possible parameters
	possibleTargetColumns = getPossibleTargetColumns()
	possibleClassifiers = getPossibleClassifiers()
	possiblePrograms = [util.getPathName(program) for program in util.getPrograms('{}/Programs'.format(os.getcwd()))]

	# Parameters
	targetColumn = None
	classifier = None
	columnsToDrop = []
	columnsToAdd = []
	program = None
	programByProgram = False
	executeWithBestParameter = False

	# Trought into all parameters
	for iCount in range(1, len(arguments), 1):
		arg = arguments[iCount]
		if arg == '--column':
			targetColumn = arguments[iCount + 1]
		elif arg == '--classifier':
			classifier = arguments[iCount + 1]
		elif arg == '--program':
			program = arguments[iCount + 1]
		elif arg == '--pbp':
			programByProgram = True
		elif arg == '--best':
			executeWithBestParameter = True

	# Set the best parameter if it is necessary
	parameter = bestParameter(targetColumn, classifier) if executeWithBestParameter else None

	if len(arguments) > 1:
		if arguments[1] == '--all': # Verify if it is for execute all classifiers with all classifications
			executeAll(possibleTargetColumns, possibleClassifiers, parameter, executeWithBestParameter=executeWithBestParameter)
			return
		elif arguments[1] == '--allPbP': #Verify if it is for execute all, but program a program
			executeAllEachProgram(possibleTargetColumns, possibleClassifiers, possiblePrograms, executeWithBestParameter)
			return

	withoutColumnMessage = 'Please specify the target column throught --column {targetColumn}. The {targetColumn} could be ' + str(possibleTargetColumns)
	withoutClassifierMessage = 'Please specify the classifier throught --classifier {classifier}. The {classifier} could be ' + str(possibleClassifiers)
	withoutProgramMessage = 'Please specify the program correctly. The {program} could be ' + str(possiblePrograms)
	errorMessage = ''
	if targetColumn is None or not targetColumn in possibleTargetColumns:
		errorMessage = '{}{}\n'.format(errorMessage, withoutColumnMessage)

	if classifier is None:
		errorMessage = '{}{}\n'.format(errorMessage, withoutClassifierMessage)
	
	if not program is None and not program in possiblePrograms:
		errorMessage = '{}{}\n'.format(errorMessage, withoutProgramMessage)

	if len(errorMessage) > 0:
		print(errorMessage)
		return

	# Execute cross validation
	if not programByProgram:
		crossValidation(targetColumn, classifier, program, columnsToDrop, columnsToAdd, parameter=parameter)
	else:
		for specifiedProgram in possiblePrograms:
			crossValidation(targetColumn, classifier, specifiedProgram, columnsToDrop, columnsToAdd, parameter=parameter)

def classify_main(arguments):
	'''
		Function responsible for receiving a mutant dataset and classifying those mutants as minimal, equivalent or traditional.
	'''
	# Possible parameters
	possibleTargetColumns = getPossibleTargetColumns()
	possibleClassifiers = getPossibleClassifiers()
	possiblePrograms = [util.getPathName(program) for program in util.getPrograms('{}/Programs'.format(os.getcwd()))]

	# Parameters
	targetColumn = None
	allTargetColumns = False
	programToClassify = None
	classifier = None
	algorithmParameter = None
	executeAllPrograms = False
	executeBestClassifierForProgram = False
	programsBestClassifiers = None
	executeAllClassifiers = False

	# Trought into all parameters
	for iCount in range(1, len(arguments), 1):
		arg = arguments[iCount]
		if arg == '--column':
			targetColumn = arguments[iCount + 1]
		elif arg == '--allColumns':
			allTargetColumns = True
		elif arg == '--program':
			programToClassify = arguments[iCount + 1]
		elif arg == '--allPrograms':
			executeAllPrograms = True
		elif arg == '--classifier':
			classifier = arguments[iCount + 1]
		elif arg == '--bestClassifier':
			executeBestClassifierForProgram = True
			programsBestClassifiers = analyzes.getBestClassifierForPrograms()
		elif arg == '--allClassifiers':
			executeAllClassifiers = True
		elif arg == '--parameter':
			algorithmParameter = int(arguments[iCount + 1])

	withoutProgramMessage = 'Please specify the program correctly. The {program} could be ' + str(possiblePrograms)
	withoutColumnMessage = 'Please specify the target column throught --column {targetColumn}. The {targetColumn} could be ' + str(possibleTargetColumns)
	withoutClassiferMessage = 'Please specify the classifier to be used throught --classifier {classifier}. The {classifier} could be ' + str(possibleClassifiers)
	errorMessage = ''

	if (targetColumn is None or not targetColumn in possibleTargetColumns) and allTargetColumns == False:
		errorMessage = '{}{}\n'.format(errorMessage, withoutColumnMessage)

	if programToClassify is None and executeAllPrograms == False:
		errorMessage = '{}{}\n'.format(errorMessage, withoutProgramMessage)

	if classifier is None and executeBestClassifierForProgram == False and executeAllClassifiers == False:
		errorMessage = '{}{}\n'.format(errorMessage, withoutClassiferMessage)

	if len(errorMessage) > 0:
		print(errorMessage)
		return

	if executeAllPrograms:
		programsToBeClassified = possiblePrograms.copy()
	else:
		programsToBeClassified = [programToClassify]

	if allTargetColumns:
		targetColumns = possibleTargetColumns.copy()
	else:
		targetColumns = [targetColumn]

	for column in targetColumns:
		for program in programsToBeClassified:
			if executeBestClassifierForProgram:
				classifier, _ = programsBestClassifiers['{}_{}'.format(program, column)]
			
			if executeAllClassifiers:
				classifiers = possibleClassifiers
			else:
				classifiers = [classifier]

			for _classifier in classifiers:
				if _classifier == 'SVM' or _classifier == 'LDA' or _classifier == 'LR' or _classifier == 'GNB':
					algorithmParameter = None
				elif _classifier == 'KNN' and column == 'MINIMAL':
					algorithmParameter = 5
				elif _classifier == 'KNN' and column == 'EQUIVALENT':
					algorithmParameter = 3
				elif _classifier == 'DT' and column == 'MINIMAL':
					algorithmParameter = 15
				elif _classifier == 'DT' and column == 'EQUIVALENT':
					algorithmParameter = 15
				elif _classifier == 'RF' and column == 'MINIMAL':
					algorithmParameter = 5
				elif _classifier == 'RF' and column == 'EQUIVALENT':
					algorithmParameter = 5

				complementClassifierName = '_{}'.format(_classifier) if executeAllClassifiers else ''
				newDataSetFileName = '{}/ML/Dataset/{}/Programs/{}.csv'.format(os.getcwd(), column, program)
				resultDataSetFileName = '{}/ML/Results/{}/Classification/{}{}.csv'.format(os.getcwd(), column, program, complementClassifierName)

				print('Program: {} | Column: {} | Classifier: {} | Parameter: {}'.format(program, column, _classifier, algorithmParameter))
				classify(newDataSetFileName, resultDataSetFileName, column, _classifier, algorithmParameter, program)

if __name__ == '__main__':
	#debug_main(sys.argv)
	classify_main(sys.argv)
	sys.exit()