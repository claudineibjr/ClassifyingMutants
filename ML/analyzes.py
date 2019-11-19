# --------------
# --- Statistics
from statistics import mean
from statistics import median

# ------
# --- OS
import os

# --------------------
# --- Machine Learning
from ML_Mutants import getPossibleClassifiers, getFullNamePossibleClassifiers

# ----------
# --- Pandas
import pandas as pd

# ---------
# --- NumPy
import numpy as np

# ---------------
# --- Matplot Lib
import matplotlib.pyplot as plt


# --------
# --- Util
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import util

import itertools

def getProgramsInfo():
	programsInfoFileName = '{}/Programs/ProgramsInfo.csv'.format(os.getcwd())
	programsInfo = util.getContentFromFile(programsInfoFileName)

	programsInfo_dict = dict()

	columns = [line.split(',') for line in programsInfo.splitlines() if len(line.split(',')[0]) > 0 and line.split(',')[0] != 'Program']
	for program in columns:
		programsInfo_dict[program[0]] = program[1:]

	return programsInfo_dict

def getProgramsHeader():
	programsInfoFileName = '{}/Programs/ProgramsInfo.csv'.format(os.getcwd())
	programsInfo = util.getContentFromFile(programsInfoFileName)

	columns = [line.split(',') for line in programsInfo.splitlines()][ : 1]
	
	return columns[0]

def getProgramInfo(programsInfo, programName):
	if programName in programsInfo.keys():
		return programsInfo[programName]

	return None

def analyzeResults(targetColumn, classifier, metric = 'F1', findTheBest = True, bestIndex = 0):
	'''
		Analyze each of 30 run for each classifier with each target column and calculate the best (calculating the mean) metric for each ones (classifier and target column).
		Returns the index of the best metric value in the due target column and classifier and the mean value of the best metric in the 30 runs
	'''
	baseResultsFolderName = '{}/ML/Results'.format(os.getcwd())

	initialCSVLineNumber = 6
	if metric == 'F1':
		columnMetrics = 4
	elif metric == 'Accuracy':
		columnMetrics = 1
	elif metric == 'Precision':
		columnMetrics = 2
	elif metric == 'Recall':
		columnMetrics = 3
	else:
		return

	# Dictionary containing the key as the index and the value a list with all metric score
	valuesMetric = dict()

	# Walks into all 30 runs
	for iCount in range(1, 30 + 1, 1):
		fileName = '{}/{}_{}/{}.csv'.format(baseResultsFolderName, targetColumn, iCount, classifier)
		
		lines = util.getContentFromFile(fileName).splitlines()
		for line in lines[initialCSVLineNumber : ]:
			columns = line.split(';')
			index = columns[0]
			valueMetric = float(columns[columnMetrics])
			
			# Append the Max metric score
			if index in valuesMetric.keys():
				valuesMetric[index].append(valueMetric)
			else:
				valuesMetric[index] = [valueMetric]
	
	# Verify the best mean Max metric score
	if findTheBest:
		indexMax, maxMetric = 0, 0
		for index, listValues in valuesMetric.items():
			if mean(listValues) > maxMetric:
				indexMax = index
				maxMetric = mean(listValues)
	else:
		indexMax = bestIndex

	# Return the best index and the due mean
	return indexMax, valuesMetric[indexMax]

def analyzeProgramAProgram(targetColumn, possibleTargetColumns, possibleClassifiers, programsInfo):
	# Verificar se precisa dessa função
	baseResultsFolderName = '{}/ML/Results/{}'.format(os.getcwd(), targetColumn)

	baseResultsProgramsFolderName = '{}/ML/Results/{}/Programs'.format(os.getcwd(), targetColumn)
	mlResults = [util.getFolderName(program) for program in util.getFilesInFolder(baseResultsProgramsFolderName)]

	programsName = [util.getFolderName(program) for program in util.getFoldersInFolder('{}/Programs'.format(os.getcwd()))]
	programsName.sort()
	
	for classifier in possibleClassifiers:
		minAccuracy = []
		minPrecision = []
		minRecall = []
		minF1 = []

		maxAccuracy = []
		maxPrecision = []
		maxRecall = []
		maxF1 = []

		meanAccuracy = []
		meanPrecision = []
		meanRecall = []
		meanF1 = []

		medianAccuracy = []
		medianPrecision = []
		medianRecall = []
		medianF1 = []

		programData = []

		for program in programsName:
			fileName = '{}_{}.csv'.format(program, classifier)
			if fileName in mlResults:
				contentFile = util.getContentFromFile('{}/{}'.format(baseResultsProgramsFolderName, fileName))

				# Separa o arquivo de resultados em colunas
				columns = [line.split(';') for line in contentFile.splitlines()]
				
				minAccuracy.append(		float(columns[0][1]) )
				minPrecision.append(	float(columns[0][2]) )
				minRecall.append(		float(columns[0][3]) )
				minF1.append(			float(columns[0][4]) )
				
				maxAccuracy.append(		float(columns[1][1]) )
				maxPrecision.append(	float(columns[1][2]) )
				maxRecall.append(		float(columns[1][3]) )
				maxF1.append(			float(columns[1][4]) )
				
				meanAccuracy.append(	float(columns[2][1]) )
				meanPrecision.append(	float(columns[2][2]) )
				meanRecall.append(		float(columns[2][3]) )
				meanF1.append(			float(columns[2][4]) )
				
				medianAccuracy.append(	float(columns[3][1]) )
				medianPrecision.append(	float(columns[3][2]) )
				medianRecall.append(	float(columns[3][3]) )
				medianF1.append(		float(columns[3][4]) )

				accuracy = float(columns[1][1])
				precision = float(columns[1][1])
				recall = float(columns[1][1])
				f1 = float(columns[1][1])
				programData.append([program, accuracy, precision, recall, f1])

		fileName = '{}/Programs_{}.csv'.format(baseResultsFolderName, classifier)
		header = ['FileName', 'Accuracy', 'Precision', 'Recall', 'F1']
		computeData(fileName, header, programData, maxAccuracy, maxPrecision, maxRecall, maxF1)

def computeData(resultsFileName, header, data, accuracy, precision, recall, f1):
	newData = []

	if len(data) < 1:
		return

	# Minímo
	subData = []
	subData.append('Min')
	subData.append(min(accuracy))   # Accuracy
	subData.append(min(precision))  # Precision
	subData.append(min(recall))     # Recall
	subData.append(min(f1))         # F1
	newData.append(subData)

	# Máximo
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

def analyzeRuns(possibleTargetColumns, possibleClassifiers):
	data = []
	
	for targetColumn in possibleTargetColumns:
		for classifier in possibleClassifiers:
			statisticsF1 = []
			stats_accuracy = []
			stats_precision = []
			stats_recall = []
			
			index, bestF1 = analyzeResults(targetColumn, classifier, 'F1')
			
			statisticsF1.append((	min(bestF1),	max(bestF1),	mean(bestF1),	median(bestF1)))

			_, accuracies = analyzeResults(targetColumn, classifier, 'Accuracy', False, index)
			stats_accuracy.append((min(accuracies), max(accuracies), mean(accuracies), median(accuracies)))

			_, precisions = analyzeResults(targetColumn, classifier, 'Precision', False, index)
			stats_precision.append((min(precisions), max(precisions), mean(precisions), median(precisions)))

			_, recalls = analyzeResults(targetColumn, classifier, 'Recall', False, index)
			stats_recall.append((min(recalls), max(recalls), mean(recalls), median(recalls)))

			subData = [targetColumn, classifier, index]
			for f1 in statisticsF1:	subData.append(f1)
			for accuracy in stats_accuracy:	subData.append(accuracy)
			for precision in stats_precision:	subData.append(precision)
			for recall in stats_recall:	subData.append(recall)

			data.append(subData)
	
	header = ['Column', 'Classifier', 'BestParameter', 'F1_Min', 'F1_Max', 'F1_Mean', 'F1_Median', 'Accuracy_Min', 'Accuracy_Max', 'Accuracy_Mean', 'Accuracy_Median', 'Precision_Min', 'Precision_Max', 'Precision_Mean', 'Precision_Median', 'Recall_Min', 'Recall_Max', 'Recall_Mean', 'Recall_Median']
	
	fileName = '{}/ML/Results/Summary/Classifiers_Statistics.csv'.format(os.getcwd())
	util.writeInCsvFile(fileName, data, header)

def analyzeExecutions():
	# TODO - Adjust for include new classifiers
	baseFolder = '{}/ML/Results/\'COLUMN\''.format(os.getcwd())
	targetColumn = ['MINIMAL', 'EQUIVALENT']
	classifiers = getPossibleClassifiers()

	equi_KNN_Accuracy = []
	equi_DT_Accuracy = []
	equi_RF_Accuracy = []
	equi_SVM_Accuracy = []
	equi_LDA_Accuracy = []
	equi_LR_Accuracy = []
	equi_GNB_Accuracy = []

	equi_KNN_Precision = []
	equi_DT_Precision = []
	equi_RF_Precision = []
	equi_SVM_Precision = []
	equi_LDA_Precision = []
	equi_LR_Precision = []
	equi_GNB_Precision = []

	equi_KNN_Recall = []
	equi_DT_Recall = []
	equi_RF_Recall = []
	equi_SVM_Recall = []
	equi_LDA_Recall = []
	equi_LR_Recall = []
	equi_GNB_Recall = []

	equi_KNN_F1 = []
	equi_DT_F1 = []
	equi_RF_F1 = []
	equi_SVM_F1 = []
	equi_LDA_F1 = []
	equi_LR_F1 = []
	equi_GNB_F1 = []

	mini_KNN_Accuracy = []
	mini_DT_Accuracy = []
	mini_RF_Accuracy = []
	mini_SVM_Accuracy = []
	mini_LDA_Accuracy = []
	mini_LR_Accuracy = []
	mini_GNB_Accuracy = []
	
	mini_KNN_Precision = []
	mini_DT_Precision = []
	mini_RF_Precision = []
	mini_SVM_Precision = []
	mini_LDA_Precision = []
	mini_LR_Precision = []
	mini_GNB_Precision = []

	mini_KNN_Recall = []
	mini_DT_Recall = []
	mini_RF_Recall = []
	mini_SVM_Recall = []
	mini_LDA_Recall = []
	mini_LR_Recall = []
	mini_GNB_Recall = []

	mini_KNN_F1 = []
	mini_DT_F1 = []
	mini_RF_F1 = []
	mini_SVM_F1 = []
	mini_LDA_F1 = []
	mini_LR_F1 = []
	mini_GNB_F1 = []

	for iCount in range(30):
		for column in targetColumn:
			folder = '{}_{}'.format(baseFolder.replace('\'COLUMN\'', column), iCount + 1)
			
			for classifier in classifiers:
				file = '{}/{}.csv'.format(folder, classifier)
				results = util.getContentFromFile(file)
				
				# Separa o arquivo de resultados em colunas
				columns = [line.split(';') for line in results.splitlines()]

				accuracy = float(	columns[1][1])
				precision = float(	columns[1][2])
				recall = float(		columns[1][3])
				f1 = float(			columns[1][4])

				if column == 'MINIMAL' 		and classifier == 'KNN':
					mini_KNN_Accuracy.append(	accuracy)
					mini_KNN_Precision.append(	precision)
					mini_KNN_Recall.append(		recall)
					mini_KNN_F1.append(			f1)

				elif column == 'MINIMAL' 	and classifier == 'DT':
					mini_DT_Accuracy.append(	accuracy)
					mini_DT_Precision.append(	precision)
					mini_DT_Recall.append(		recall)
					mini_DT_F1.append(			f1)
				
				elif column == 'MINIMAL' 	and classifier == 'RF':
					mini_RF_Accuracy.append(	accuracy)
					mini_RF_Precision.append(	precision)
					mini_RF_Recall.append(		recall)
					mini_RF_F1.append(			f1)

				elif column == 'MINIMAL' 	and classifier == 'SVM':
					mini_SVM_Accuracy.append(	accuracy)
					mini_SVM_Precision.append(	precision)
					mini_SVM_Recall.append(		recall)
					mini_SVM_F1.append(			f1)

				elif column == 'MINIMAL' 	and classifier == 'LDA':
					mini_LDA_Accuracy.append(	accuracy)
					mini_LDA_Precision.append(	precision)
					mini_LDA_Recall.append(		recall)
					mini_LDA_F1.append(			f1)

				elif column == 'MINIMAL' 	and classifier == 'LR':
					mini_LR_Accuracy.append(	accuracy)
					mini_LR_Precision.append(	precision)
					mini_LR_Recall.append(		recall)
					mini_LR_F1.append(			f1)

				elif column == 'MINIMAL' 	and classifier == 'GNB':
					mini_GNB_Accuracy.append(	accuracy)
					mini_GNB_Precision.append(	precision)
					mini_GNB_Recall.append(		recall)
					mini_GNB_F1.append(			f1)
				
				elif column == 'EQUIVALENT'	and classifier == 'KNN':
					equi_KNN_Accuracy.append(	accuracy)
					equi_KNN_Precision.append(	precision)
					equi_KNN_Recall.append(		recall)
					equi_KNN_F1.append(			f1)
				
				elif column == 'EQUIVALENT'	and classifier == 'DT':
					equi_DT_Accuracy.append(	accuracy)
					equi_DT_Precision.append(	precision)
					equi_DT_Recall.append(		recall)
					equi_DT_F1.append(			f1)
				
				elif column == 'EQUIVALENT'	and classifier == 'RF':
					equi_RF_Accuracy.append(	accuracy)
					equi_RF_Precision.append(	precision)
					equi_RF_Recall.append(		recall)
					equi_RF_F1.append(			f1)

				elif column == 'EQUIVALENT'	and classifier == 'SVM':
					equi_SVM_Accuracy.append(	accuracy)
					equi_SVM_Precision.append(	precision)
					equi_SVM_Recall.append(		recall)
					equi_SVM_F1.append(			f1)

				elif column == 'EQUIVALENT'	and classifier == 'LDA':
					equi_LDA_Accuracy.append(	accuracy)
					equi_LDA_Precision.append(	precision)
					equi_LDA_Recall.append(		recall)
					equi_LDA_F1.append(			f1)

				elif column == 'EQUIVALENT'	and classifier == 'LR':
					equi_LR_Accuracy.append(	accuracy)
					equi_LR_Precision.append(	precision)
					equi_LR_Recall.append(		recall)
					equi_LR_F1.append(			f1)

				elif column == 'EQUIVALENT'	and classifier == 'GNB':
					equi_GNB_Accuracy.append(	accuracy)
					equi_GNB_Precision.append(	precision)
					equi_GNB_Recall.append(		recall)
					equi_GNB_F1.append(			f1)

	baseFolder = '{}/ML/Results/Summary'.format(os.getcwd())
	if not util.pathExists(baseFolder):
		util.createFolder(baseFolder)

	header = ['Accuracy', 'Precision', 'Recall', 'F1']

	# Equivalent - KNN
	data = [equi_KNN_Accuracy, equi_KNN_Precision, equi_KNN_Recall, equi_KNN_F1]
	writeData('{}/EQUIVALENT_KNN.csv'.format(baseFolder), data, header)

	# Equivalent - Decision Tree
	data = [equi_DT_Accuracy, equi_DT_Precision, equi_DT_Recall, equi_DT_F1]
	writeData('{}/EQUIVALENT_DT.csv'.format(baseFolder), data, header)

	# Equivalent - Random Forest
	data = [equi_RF_Accuracy, equi_RF_Precision, equi_RF_Recall, equi_RF_F1]
	writeData('{}/EQUIVALENT_RF.csv'.format(baseFolder), data, header)

	# Equivalent - Support Vector Machine
	data = [equi_SVM_Accuracy, equi_SVM_Precision, equi_SVM_Recall, equi_SVM_F1]
	writeData('{}/EQUIVALENT_SVM.csv'.format(baseFolder), data, header)

	# Equivalent - Linear Discriminant Analysis
	data = [equi_LDA_Accuracy, equi_LDA_Precision, equi_LDA_Recall, equi_LDA_F1]
	writeData('{}/EQUIVALENT_LDA.csv'.format(baseFolder), data, header)

	# Equivalent - Logistic Regression
	data = [equi_LR_Accuracy, equi_LR_Precision, equi_LR_Recall, equi_LR_F1]
	writeData('{}/EQUIVALENT_LR.csv'.format(baseFolder), data, header)

	# Equivalent - Gaussian Naive Bayes
	data = [equi_GNB_Accuracy, equi_GNB_Precision, equi_GNB_Recall, equi_GNB_F1]
	writeData('{}/EQUIVALENT_GNB.csv'.format(baseFolder), data, header)

	# Minimals - KNN
	data = [mini_KNN_Accuracy, mini_KNN_Precision, mini_KNN_Recall, mini_KNN_F1]
	writeData('{}/MINIMAL_KNN.csv'.format(baseFolder), data, header)

	# Minimals - Decision Tree
	data = [mini_DT_Accuracy, mini_DT_Precision, mini_DT_Recall, mini_DT_F1]
	writeData('{}/MINIMAL_DT.csv'.format(baseFolder), data, header)

	# Minimals - Random Forest
	data = [mini_RF_Accuracy, mini_RF_Precision, mini_RF_Recall, mini_RF_F1]
	writeData('{}/MINIMAL_RF.csv'.format(baseFolder), data, header)

	# Minimals - Support Vector Machine
	data = [equi_SVM_Accuracy, equi_SVM_Precision, equi_SVM_Recall, equi_SVM_F1]
	writeData('{}/MINIMAL_SVM.csv'.format(baseFolder), data, header)

	# Minimals - Linear Discriminant Analysis
	data = [equi_LDA_Accuracy, equi_LDA_Precision, equi_LDA_Recall, equi_LDA_F1]
	writeData('{}/MINIMAL_LDA.csv'.format(baseFolder), data, header)

	# Minimals - Logistic Regression
	data = [equi_LR_Accuracy, equi_LR_Precision, equi_LR_Recall, equi_LR_F1]
	writeData('{}/MINIMAL_LR.csv'.format(baseFolder), data, header)

	# Minimals - Gaussian Naive Bayes
	data = [equi_GNB_Accuracy, equi_GNB_Precision, equi_GNB_Recall, equi_GNB_F1]
	writeData('{}/MINIMAL_GNB.csv'.format(baseFolder), data, header)

def writeData(fileName, data, header):
	# Verificar
	newData = []

	accuracy = data[0]
	precision = data[1]
	recall = data[2]
	f1 = data[3]

	subData = []
	subData.append(max(accuracy))
	subData.append(max(precision))
	subData.append(max(recall))
	subData.append(max(f1))
	newData.append(subData)

	newData.append(header)
	newData.append('')
	for iCount in range(len(accuracy)):
		tempData = [accuracy[iCount], precision[iCount], recall[iCount], f1[iCount]]
		newData.append(tempData)

	util.writeInCsvFile(fileName, newData)

def bestParameterFileExists(file):
	fileFilter = '_bestParameter'
	if file.__contains__(fileFilter):
		if util.pathExists(file):
			return file
		else:
			return file.replace(fileFilter, '')
	else:
		return file

def getMetricsFromPrograms(possibleTargetColumns, possibleClassifiers, programsInfo, writeMetrics = False, bestParameter = False):
	fileFilter = '_bestParameter' if bestParameter else ''
	programs = [util.getFolderName(program) for program in util.getPrograms('{}/Programs'.format(os.getcwd()))]

	i_program_Max = 1
	#i_program_Accuracy = 1
	#i_program_Precision = 2
	#i_program_Recall = 3
	i_program_F1 = 4

	programsHeader = getProgramsHeader()
	
	columnsHeader = programsHeader.copy()
	columnsHeader.remove('Program')
	df_programsInfo = pd.DataFrame.from_dict(programsInfo, orient='index', columns=columnsHeader)

	# Column label on CSV programs info
	# MM_RF_F1
	# MM_DT_F1
	# MM_KNN_F1
	# MM_SVM_F1
	# MM_LDA_F1
	# MM_LR_F1
	# MM_GNB_F1
	# EM_RF_F1
	# EM_DT_F1
	# EM_KNN_F1
	# EM_SVM_F1
	# EM_LDA_F1
	# EM_LR_F1
	# EM_GNB_F1

	for program in programs:

		# Split the file in lines and columns (;)
		fileName = '{}/ML/Results/MINIMAL/Programs/{}_[CLASSIFIER]{}.csv'.format(os.getcwd(), program, fileFilter)
		file_Minimal_RF = util.splitFileInColumns(	bestParameterFileExists(fileName.replace('[CLASSIFIER]', 'RF') ), ';')
		file_Minimal_DT = util.splitFileInColumns(	bestParameterFileExists(fileName.replace('[CLASSIFIER]', 'DT') ), ';')
		file_Minimal_kNN = util.splitFileInColumns(	bestParameterFileExists(fileName.replace('[CLASSIFIER]', 'KNN')), ';')
		file_Minimal_SVM = util.splitFileInColumns(	bestParameterFileExists(fileName.replace('[CLASSIFIER]', 'SVM')), ';')
		file_Minimal_LDA = util.splitFileInColumns(	bestParameterFileExists(fileName.replace('[CLASSIFIER]', 'LDA')), ';')
		file_Minimal_LR = util.splitFileInColumns(	bestParameterFileExists(fileName.replace('[CLASSIFIER]', 'LR') ), ';')
		file_Minimal_GNB = util.splitFileInColumns(	bestParameterFileExists(fileName.replace('[CLASSIFIER]', 'GNB')), ';')
		
		fileName = '{}/ML/Results/EQUIVALENT/Programs/{}_[CLASSIFIER]{}.csv'.format(os.getcwd(), program, fileFilter)
		file_Equivalent_RF = util.splitFileInColumns(	bestParameterFileExists(fileName.replace('[CLASSIFIER]', 'RF') ), ';')
		file_Equivalent_DT = util.splitFileInColumns(	bestParameterFileExists(fileName.replace('[CLASSIFIER]', 'DT') ), ';')
		file_Equivalent_kNN = util.splitFileInColumns(	bestParameterFileExists(fileName.replace('[CLASSIFIER]', 'KNN')), ';')
		file_Equivalent_SVM = util.splitFileInColumns(	bestParameterFileExists(fileName.replace('[CLASSIFIER]', 'SVM')), ';')
		file_Equivalent_LDA = util.splitFileInColumns(	bestParameterFileExists(fileName.replace('[CLASSIFIER]', 'LDA')), ';')
		file_Equivalent_LR = util.splitFileInColumns(	bestParameterFileExists(fileName.replace('[CLASSIFIER]', 'LR') ), ';')
		file_Equivalent_GNB = util.splitFileInColumns(	bestParameterFileExists(fileName.replace('[CLASSIFIER]', 'GNB')), ';')

		# Update the metrics of programs info
		df_programsInfo.loc[program]['MM_RF_F1'] = file_Minimal_RF[i_program_Max][i_program_F1]
		df_programsInfo.loc[program]['MM_DT_F1'] = file_Minimal_DT[i_program_Max][i_program_F1]
		df_programsInfo.loc[program]['MM_KNN_F1'] = file_Minimal_kNN[i_program_Max][i_program_F1]
		df_programsInfo.loc[program]['MM_SVM_F1'] = file_Minimal_SVM[i_program_Max][i_program_F1]
		df_programsInfo.loc[program]['MM_LDA_F1'] = file_Minimal_LDA[i_program_Max][i_program_F1]
		df_programsInfo.loc[program]['MM_LR_F1'] = file_Minimal_LR[i_program_Max][i_program_F1]
		df_programsInfo.loc[program]['MM_GNB_F1'] = file_Minimal_GNB[i_program_Max][i_program_F1]

		df_programsInfo.loc[program]['EM_RF_F1'] = file_Equivalent_RF[i_program_Max][i_program_F1]
		df_programsInfo.loc[program]['EM_DT_F1'] = file_Equivalent_DT[i_program_Max][i_program_F1]
		df_programsInfo.loc[program]['EM_KNN_F1'] = file_Equivalent_kNN[i_program_Max][i_program_F1]
		df_programsInfo.loc[program]['EM_SVM_F1'] = file_Equivalent_SVM[i_program_Max][i_program_F1]
		df_programsInfo.loc[program]['EM_LDA_F1'] = file_Equivalent_LDA[i_program_Max][i_program_F1]
		df_programsInfo.loc[program]['EM_LR_F1'] = file_Equivalent_LR[i_program_Max][i_program_F1]
		df_programsInfo.loc[program]['EM_GNB_F1'] = file_Equivalent_GNB[i_program_Max][i_program_F1]
	
	if (writeMetrics):
		# Writting program info
		programsInfoFileName = '{}/Programs/ProgramsInfo.csv'.format(os.getcwd())
		data = []
		data.append(programsHeader)
		for index, values in df_programsInfo.iterrows():
			values = list(values.values)
			values.insert(0, index)
			data.append(values)
		util.writeInCsvFile(programsInfoFileName, data, delimiter = ',')

	return df_programsInfo

def analyzeMetricsFromProgram(metricsFromProgram):
	'''
		Function responsible to verify the programs and identify result of the best classifier
	'''

	# Remove useless columns
	metricsFromProgram = metricsFromProgram.drop(['', 'Functions', 'Line of Code', 'Mutants', 'Minimals', '%', 'Equivalents', 'Test Cases'], axis=1)


	minimalMetrics = metricsFromProgram.drop(['EM_RF_F1', 'EM_DT_F1', 'EM_KNN_F1', 'EM_SVM_F1', 'EM_LDA_F1', 'EM_LR_F1', 'EM_GNB_F1'], axis = 1)
	equivalentMetrics = metricsFromProgram.drop(['MM_RF_F1', 'MM_DT_F1', 'MM_KNN_F1', 'MM_SVM_F1', 'MM_LDA_F1', 'MM_LR_F1', 'MM_GNB_F1'], axis = 1)

	# Iter trough Dataframes and add the max value for each program in the dictionary
	programsBestMetrics = dict()
	for program, values in minimalMetrics.iterrows():
		programsBestMetrics[program] = [max(values), 0]
	for program, values in equivalentMetrics.iterrows():
		programsBestMetrics[program] = [programsBestMetrics[program][0], max(values)]

	# Create a dataframe, where the index is the program and has two columns, max F1 for minimals and max f1 for equivalent
	programsBestMetrics = pd.DataFrame.from_dict(programsBestMetrics, orient='index', columns=['MINIMAL', 'EQUIVALENT'])

	return programsBestMetrics

if __name__ == '__main__':
	programsInfo = getProgramsInfo()
	
	possibleTargetColumns = ['MINIMAL', 'EQUIVALENT']
	possibleClassifiers = getPossibleClassifiers()

	# --- Analyze the 30 runs and calc statistics informations, like minimum, maximum, median and average
	#analyzeRuns(possibleTargetColumns, possibleClassifiers)

	# ------------------------------------------------------
	# --- Get informations and write ML metrics for programs
	#metrics = getMetricsFromPrograms(possibleTargetColumns, possibleClassifiers, programsInfo, bestParameter=True)
	#programsBestMetrics = analyzeMetricsFromProgram(metrics)

	#analyzeClassifiers(metrics, possibleClassifiers, plot=True)

	# -------------------------------------------
	# --- Analyze the executions for each program
	#analyzeProgramAProgram('MINIMAL', possibleTargetColumns, possibleClassifiers, programsInfo)
	#analyzeProgramAProgram('EQUIVALENT', possibleTargetColumns, possibleClassifiers, programsInfo)