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

def analyzeRuns(possibleTargetColumns, possibleClassifiers, plot = False, writeFile = False):
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
	
	if writeFile:
		header = ['Column', 'Classifier', 'BestParameter', 'F1_Min', 'F1_Max', 'F1_Mean', 'F1_Median', 'Accuracy_Min', 'Accuracy_Max', 'Accuracy_Mean', 'Accuracy_Median', 'Precision_Min', 'Precision_Max', 'Precision_Mean', 'Precision_Median', 'Recall_Min', 'Recall_Max', 'Recall_Mean', 'Recall_Median']
		
		fileName = '{}/ML/Results/Summary/Classifiers_Statistics.csv'.format(os.getcwd())
		util.writeInCsvFile(fileName, data, header, delimiter=',')

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

def analyzeMetricsFromProgram(metricsFromProgram, possibleClassifiers, plot = False):
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
		maxValue = max(values)
		maxIndex = list(values).index(maxValue)
		bestColumn = str(values.index[maxIndex]).replace('MM_', '').replace('_F1', '')
		
		programsBestMetrics[program] = [maxValue, bestColumn, None, None]
	for program, values in equivalentMetrics.iterrows():
		maxValue = max(values)
		maxIndex = list(values).index(maxValue)
		bestColumn = str(values.index[maxIndex]).replace('EM_', '').replace('_F1', '')
		
		programsBestMetrics[program] = [programsBestMetrics[program][0], programsBestMetrics[program][1], maxValue, bestColumn]

	# Create a dataframe, where the index is the program and has five columns
	# 	Program name, Max F1 for minimals, Classifier that achieves the best F1 for minimals, Max f1 for equivalents and Classifier that achieves the best F1 for equivalents
	df_programsBestMetrics = pd.DataFrame()

	for program, value in programsBestMetrics.items():
		data = [ program, value[0], value[1], value[2], value[3] ]
		columns = ['Program', 'MINIMAL', 'MM_Classifier', 'EQUIVALENT', 'EM_Classifier']
		newDataFrame = pd.DataFrame(data=[data], columns=columns)
		df_programsBestMetrics = df_programsBestMetrics.append(newDataFrame)

	if plot:
		plotMetricsFromProgram(df_programsBestMetrics, possibleClassifiers)

	return df_programsBestMetrics

def plotMetricsFromProgram(programsBestMetrics, possibleClassifiers):
	# Set lists with data. For each array, there are several internal arrays containing the F1 values for each program
	dataMinimal = []
	dataEquivalent = []
	for iCount in range(len(possibleClassifiers)):
		dataMinimal.append(0)
		dataEquivalent.append(0)
	
	for row in programsBestMetrics.itertuples():
		MM_classifier = getattr(row, 'MM_Classifier')
		dataMinimal[possibleClassifiers.index(MM_classifier)] = dataMinimal[possibleClassifiers.index(MM_classifier)] + 1

		EM_classifier = getattr(row, 'EM_Classifier')
		dataEquivalent[possibleClassifiers.index(EM_classifier)] = dataEquivalent[possibleClassifiers.index(EM_classifier)] + 1

	# Create the figure with axis
	fig = plt.figure(1, figsize=(9, 6))
	ax = fig.add_subplot(1, 1, 1)

	# Set the value to be shown as indexes on axis Y
	#ax.set_yticks([value for value in range(0, 101, 10)])
	ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.75)

	# Set the chart title and axis title
	#ax.set_title('axes title', fontsize = 14)
	ax.set_xlabel('\nClassifiers', fontsize = 14)
	ax.set_ylabel('Number of programs that the classifier was the best', fontsize = 14)

	# Set the boxplot positions
	width = 0.4  # the width of the bars
	positionsMM = [value - width / 2 for value in range(len(possibleClassifiers))]
	positionsEM = [value + width / 2 for value in range(len(possibleClassifiers))]

	# Set the bar chart based on a function property defined below
	bcMM = barChartProperties(ax, dataMinimal, positionsMM, '#5975a4', width)
	bcEM = barChartProperties(ax, dataEquivalent, positionsEM, '#b55d60', width)

	# Set the label between two boxplot
	ax.set_xticklabels(possibleClassifiers)
	ax.set_xticks([value for value in range(len(possibleClassifiers))])

	# Set the chart subtitle/legend
	ax.legend([bcMM, bcEM], ['Minimal Mutants', 'Equivalent Mutants'], loc='upper left')

	autolabel(bcMM, ax)
	autolabel(bcEM, ax)

	fig.tight_layout()

	# Display chart
	plt.show()

def barChartProperties(ax, data, positions, color, width):
	barChart = ax.bar(positions, data, width=width, color=color)
	
	return barChart
	
def autolabel(rects, ax):
	"""Attach a text label above each bar in *rects*, displaying its height."""
	for rect in rects:
		height = rect.get_height()
		ax.annotate('{}'.format(height),
					xy=(rect.get_x() + rect.get_width() / 2, height),
					xytext=(0, 3),  # 3 points vertical offset
					textcoords="offset points",
					ha='center', va='bottom')

def analyzeClassifiersProgramAProgram(metricsFromProgram, possibleClassifiers, plot=False):
	'''
		Function responsible to verify the classifiers and indicate the results program by program
	'''

	# Remove useless columns
	metricsFromProgram = metricsFromProgram.drop(['', 'Functions', 'Line of Code', 'Mutants', 'Minimals', '%', 'Equivalents', 'Test Cases'], axis=1)
	
	# Renaming columns by removing _F1 at the end
	metricsFromProgram = metricsFromProgram.rename(lambda x: x.replace('_F1', '') ,axis='columns')

	# Each column_classifier is a value in the dictionary (Example: EM_KNN, MM_RF, etc ...)
	metricsFromClassifier = pd.DataFrame()

	for program, values in metricsFromProgram.iterrows():
		for key, value in values.iteritems():
			classifier = key[key.index('_') + 1 : ]
			column = 'EQUIVALENT' if key[ : key.index('_')] == 'EM' else 'MINIMAL'
			f1 = float(value)
			
			newDataFrame = pd.DataFrame(data=[[ classifier, column, program, f1 ]], columns=['Classifier', 'Column', 'Program', 'F1'])
			metricsFromClassifier = metricsFromClassifier.append(newDataFrame)

	if plot:
		plotClassifiersProgramAProgram(metricsFromClassifier, possibleClassifiers)

	return metricsFromClassifier

def plotClassifiersProgramAProgram(metricsFromClassifier, possibleClassifiers):
	'''
		Function responsible to show a boxplot for classifiers.
	'''

	# Set lists with data. For each array, there are several internal arrays containing the F1 values for each program
	dataMinimal = []
	dataEquivalent = []
	for classifier in possibleClassifiers:
		dataMinimal.append([])
		dataEquivalent.append([])
	
	for row in metricsFromClassifier.itertuples():
		classifier = getattr(row, 'Classifier')
		column = getattr(row, 'Column')
		
		if column == 'MINIMAL':
			dataMinimal[possibleClassifiers.index(classifier)].append(getattr(row, 'F1'))
		elif column == 'EQUIVALENT':
			dataEquivalent[possibleClassifiers.index(classifier)].append(getattr(row, 'F1'))
	
	# Create the figure with axis
	fig = plt.figure(1, figsize=(9, 6))
	ax = fig.add_subplot(1, 1, 1)

	# Set the value to be shown as indexes on axis Y
	ax.set_yticks([value for value in range(0, 101, 10)])
	ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.75)

	# Set the chart title and axis title
	#ax.set_title('axes title', fontsize = 14)
	ax.set_xlabel('\nClassifiers', fontsize = 14)
	ax.set_ylabel('F1 Score', fontsize = 14)

	# Set the boxplot positions
	positionsMM = [value for value in range(len(possibleClassifiers) * 2) if value % 2 == 0 ]
	positionsEM = [value + 0.5 for value in range(len(possibleClassifiers) * 2) if value % 2 == 0 ]

	# Set the boxplots based on a function property defined below
	bpMM = boxPlotProperties(ax, dataMinimal, positionsMM, '#5975a4')
	bpEM = boxPlotProperties(ax, dataEquivalent, positionsEM, '#b55d60')


	# Calculate the medians
	minimalMedians = [ np.round(np.median(value), 2) for value in dataMinimal]
	equivalentMedians = [np.round(np.median(value), 2) for value in dataEquivalent]

	# Calculate the means
	#minimalMeans = [ np.round(np.mean(value), 2) for value in dataMinimal]
	#equivalentMeans = [np.round(np.mean(value), 2) for value in dataEquivalent]

	# Display values in boxplot
	for tick in range(len(positionsMM)):
		# Display a median on boxplot top
		ax.text(positionsMM[tick], 101, minimalMedians[tick], horizontalalignment='center', size='small', color='black', weight='semibold')
		ax.text(positionsEM[tick], 101, equivalentMedians[tick], horizontalalignment='center', size='small', color='black', weight='semibold')

		#ax.text(positionsMM[tick], minimalMeans[tick] + 0.5, minimalMeans[tick], horizontalalignment='center', size='small', color='black', weight='semibold')
		#ax.text(positionsEM[tick], equivalentMeans[tick] + 0.5, equivalentMeans[tick], horizontalalignment='center', size='small', color='black', weight='semibold')

			
	# Set the label between two boxplot
	#ax.set_xticklabels(list(getFullNamePossibleClassifiers().values())) # Caso for exibir o nome completo do classificador
	ax.set_xticklabels(possibleClassifiers)
	ax.set_xticks([value + 0.25 for value in ax.get_xticks() if value % 2 == 0])

	# Set the chart subtitle/legend
	ax.legend([bpMM["boxes"][0], bpEM["boxes"][0]], ['Minimal Mutants', 'Equivalent Mutants'], loc='lower right')

	fig.tight_layout()

	# Display chart
	plt.show()

def boxPlotProperties(ax, data, positions, color):
	boxprops = dict(facecolor=color, linewidth = 1)
	medianprops = dict(color='#000000', linewidth = 1)
	meanprops = dict(color='#000000', linewidth = 1)

	boxplot = ax.boxplot(data, sym='+', positions=positions, patch_artist=True, meanline=True,
		showmeans=True, labels=possibleClassifiers, boxprops=boxprops, 
		medianprops=medianprops, meanprops=meanprops)

	return boxplot

if __name__ == '__main__':
	possibleTargetColumns = ['MINIMAL', 'EQUIVALENT']
	possibleClassifiers = getPossibleClassifiers()

	# ---------------------------------------------------------------------------------------------------
	# --- Analyze the 30 runs and calc statistics informations, like minimum, maximum, median and average
	analyzeRuns(possibleTargetColumns, possibleClassifiers, plot = False, writeFile = True)

	# ----------------------------------
	# --- Get informations from programs
	programsInfo = getProgramsInfo()
	programsInfo = getMetricsFromPrograms(possibleTargetColumns, possibleClassifiers, programsInfo, bestParameter=True)
	
	programsBestMetrics = analyzeMetricsFromProgram(programsInfo, possibleClassifiers, plot=False)

	analyzeClassifiersProgramAProgram(programsInfo, possibleClassifiers, plot=False)