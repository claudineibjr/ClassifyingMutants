# --------------
# --- Statistics
from statistics import mean
from statistics import median

# ------
# --- OS
import os

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

from util import getPossibleClassifiers, getFullNamePossibleClassifiers

import itertools

# --------------
# --- ML_Mutants
#from ML_Mutants import evaluatingClassification
import ML_Mutants

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

def analyzeResults(possibleTargetColumns, possibleClassifiers, overwriteFullFile = False, plot = False):
	'''
		Analyze each of 30 run for each classifier with each target column and calculate the best (calculating the mean) metric for each ones (classifier and target column).
	'''

	# Dataframe com todos os resultados das execuções
	experimentResults = pd.DataFrame()

	# Dataframe com a indicação do melhor parâmetro para cada classificador
	classifiersBestParameter = pd.DataFrame()
	
	# Base folder
	baseResultsFolderName = '{}/ML/Results'.format(os.getcwd())
	
	# FullFile
	fullFileName = '{}/Summary/Summary_All30Runs.csv'.format(baseResultsFolderName)

	# SummaryFile
	summaryFileName = '{}/Summary/Summary_Classifiers.csv'.format(baseResultsFolderName)

	if overwriteFullFile or not util.pathExists(fullFileName):
		# Caso for sobrescrever o arquivo já existente ou caso ele não exista, faz toda a leitura e escreve um novo arquivo

		# Percorre todas as colunas (Minimal e equivalent)
		for targetColumn in possibleTargetColumns:
			# Percorre todos os classificadores
			for classifier in possibleClassifiers:
				# Dataframe contendo o resultado das 30 execuções de cada um dos programas
				classifierResults = pd.DataFrame()
				
				# Percorre todas as execuções (de 1 a 30)
				for iCount in range(1, 31, 1):
					# Busca os resultados da execução
					classifierRunResultFileName = '{}/{}_{}/{}.csv'.format(baseResultsFolderName, targetColumn, iCount, classifier)
					classifierRunResult = util.createDataFrameFromCSV(classifierRunResultFileName, hasHeader=True, separator=';', initialLine=5)
					
					# Concatena o resultado dessa execução às outras execuções que no final serão 30
					classifierResults = classifierResults.append(classifierRunResult)

					# Insere as informações referentes àquela execução
					pd.DataFrame(classifierRunResult).insert(0, 'TargetColumn', targetColumn, True)
					pd.DataFrame(classifierRunResult).insert(1, 'Classifier', classifier, True)
					pd.DataFrame(classifierRunResult).insert(2, 'Run', iCount, True)

					# Concatena essa execução a todas as outras
					experimentResults = experimentResults.append(classifierRunResult)

				# Calcula o melhor parâmetro para aquele classificador
				parameters = classifierResults['SampleSplit'].unique()
				parameterMetrics = pd.DataFrame()
				for parameter in parameters:
					# Busca apenas os resultados daquele parâmetro
					resultsFromThisParameter = classifierResults.query('SampleSplit == \'{}\''.format(parameter))
					
					# Coleta a média das métricas das 30 execuções
					meanAccuracy = resultsFromThisParameter['Accuracy'].sum() / 30
					meanPrecision = resultsFromThisParameter['Precision'].sum() / 30
					meanRecall = resultsFromThisParameter['Recall'].sum() / 30
					meanF1 = resultsFromThisParameter['F1'].sum() / 30

					parameterMetrics = parameterMetrics.append(pd.DataFrame(data=[[targetColumn, classifier, parameter, meanAccuracy, meanPrecision, meanRecall, meanF1]], columns=['TargetColumn', 'Classifier', 'Parameter', 'Accuracy', 'Precision', 'Recall', 'F1']))
					#print('Parameter: {}\t\tAccuracy: {} | Precision: {} | Recall: {} | F1: {}'.format(parameter, meanAccuracy, meanPrecision, meanRecall, meanF1))
				
				bestParameter = parameterMetrics.sort_values(by=['F1'], ascending=False).head(n=1)
				#print('\n--- {} - {}'.format(classifier, targetColumn))
				#print(bestParameter.head())
				
				classifiersBestParameter = classifiersBestParameter.append(bestParameter)

		#print(classifiersBestParameter.head(n=50))

		# Escreve um arquivo só com todos os resultados
		util.writeDataFrameInCsvFile(fullFileName, experimentResults, index=False)

		# Escreve um arquivo só com os melhores parâmetros para cada classificador
		util.writeDataFrameInCsvFile(summaryFileName, classifiersBestParameter, index=False)

	else:
		# Busca os arquivos já existentes
		experimentResults = util.createDataFrameFromCSV(fullFileName, True, ',')
		classifiersBestParameter = util.createDataFrameFromCSV(summaryFileName, True, ',')

	if plot:
		plotRunsResult(experimentResults, possibleClassifiers, possibleTargetColumns)

	return experimentResults, classifiersBestParameter

def plotRunsResult(runsResults, possibleClassifiers, possibleTargetColumns, plotSeparated = False):
	'''
		Is calculated the average of each classifier/target column
	'''

	if plotSeparated:
		targetColumnsOption = {'MINIMAL': ('#607D8B'), 'EQUIVALENT': ('#B0BEC5')}
		
		for targetColumn, (color) in targetColumnsOption.items():
			targetColumnData = []
			for _ in range(len(possibleClassifiers)):
				targetColumnData.append(0)

			for classifier in possibleClassifiers:
				Values_Classifier_Column = runsResults.query('Classifier == \'{}\' and TargetColumn == \'{}\' '.format(classifier, targetColumn))

				targetColumnData[possibleClassifiers.index(classifier)] = np.mean(Values_Classifier_Column['F1'])

			# Create the figure with axis
			fig = plt.figure(1, figsize=(9, 6))
			ax = fig.add_subplot(1, 1, 1)

			# Set the value to be shown as indexes on axis Y
			ax.set_yticks([value for value in range(0, 50, 10)] + [value for value in range(50, 101, 5)])
			ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.75)

			# Set the chart title and axis title
			ax.set_title('Mean F1 Score for Classifiers', fontsize = 14)
			ax.set_xlabel('\nClassifiers', fontsize = 14)
			ax.set_ylabel('F1 Score', fontsize = 14)

			# Set the boxplot positions
			width = 0.6  # the width of the bars
			positions = [value for value in range(len(possibleClassifiers))]

			# Set the bar chart based on a function property defined below
			barChart = barChartProperties(ax, targetColumnData, positions, color, None)

			# Set the label between two boxplot
			ax.set_xticklabels(possibleClassifiers)
			ax.set_xticks([value for value in range(len(possibleClassifiers))])

			# Set the chart subtitle/legend
			#ax.legend([barChart, bcEM], ['Minimal Mutants', 'Equivalent Mutants'], loc='upper right')

			autolabel(barChart, ax, 2)

			fig.tight_layout()

			# Display chart
			plt.show()

	else:
		dataMinimal = []
		dataEquivalent = []
		for iCount in range(len(possibleClassifiers)):
			dataMinimal.append(0)
			dataEquivalent.append(0)

		for column in possibleTargetColumns:
			for classifier in possibleClassifiers:
				Values_Classifier_Column = runsResults.query('Classifier == \'{}\' and TargetColumn == \'{}\' '.format(classifier, column))

				meanAccuracy = np.mean(Values_Classifier_Column['Accuracy'])
				meanPrecision = np.mean(Values_Classifier_Column['Precision'])
				meanRecall = np.mean(Values_Classifier_Column['Recall'])
				meanF1 = np.mean(Values_Classifier_Column['F1'])

				#print('Classifier: {}\tColumn: {}\t\tAccuracy: {:.2f}\tPrecision: {:.2f}\tRecall: {:.2f}\tF1: {:.2f}'.format(classifier, column, meanAccuracy, meanPrecision, meanRecall, meanF1))

				if column == 'MINIMAL':
					dataMinimal[possibleClassifiers.index(classifier)] = meanF1
				elif column == 'EQUIVALENT':
					dataEquivalent[possibleClassifiers.index(classifier)] = meanF1

		# Create the figure with axis
		fig = plt.figure(1, figsize=(9, 6))
		ax = fig.add_subplot(1, 1, 1)

		# Set the value to be shown as indexes on axis Y
		ax.set_yticks([value for value in range(0, 50, 10)] + [value for value in range(50, 101, 5)])
		ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.75)

		# Set the chart title and axis title
		ax.set_title('Mean F1 Score for Classifiers', fontsize = 14)
		ax.set_xlabel('\nClassifiers', fontsize = 14)
		ax.set_ylabel('F1 Score', fontsize = 14)

		# Set the boxplot positions
		width = 0.3  # the width of the bars
		positionsMM = [value - width / 2 for value in range(len(possibleClassifiers))]
		positionsEM = [value + width / 2 for value in range(len(possibleClassifiers))]

		# Set the bar chart based on a function property defined below
		barChart = barChartProperties(ax, dataMinimal, positionsMM, '#607D8B', width)
		bcEM = barChartProperties(ax, dataEquivalent, positionsEM, '#B0BEC5', width)

		# Set the label between two boxplot
		ax.set_xticklabels(possibleClassifiers)
		ax.set_xticks([value for value in range(len(possibleClassifiers))])

		# Set the chart subtitle/legend
		ax.legend([barChart, bcEM], ['Minimal Mutants', 'Equivalent Mutants'], loc='upper right')

		autolabel(barChart, ax, 2)
		autolabel(bcEM, ax, 2)

		fig.tight_layout()

		# Display chart
		plt.show()

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
	programs = [util.getPathName(program) for program in util.getPrograms('{}/Programs'.format(os.getcwd()))]

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
	'''
		Function that graphs the number of programs in which each classifier was considered the best
	'''

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
	if width is None:
		barChart = ax.bar(positions, data, color=color)
	else:
		barChart = ax.bar(positions, data, width=width, color=color)
	
	return barChart
	
def autolabel(rects, ax, decimals = -1):
	"""Attach a text label above each bar in *rects*, displaying its height."""
	for iCount, rect in zip(range(len(rects)), rects):
		
		if decimals > -1:
			height = np.round(rect.get_height(), decimals)
		else:
			height = rect.get_height()
				
		ax.annotate('{}'.format(height),
					xy=(rect.get_x() + rect.get_width() / 2, height),
					xytext=(0, 3),  # 3 points vertical offset
					textcoords="offset points",
					ha='center', va='bottom')

def analyzeClassifiersProgramAProgram(metricsFromProgram, possibleClassifiers, plot=False):
	'''
		Function responsible to verify the classifiers and indicate the results program by program
		Returns a dataframe containing the classifiers, columns, ProgramsName and the F1s
	'''

	# Remove useless columns
	metricsFromProgram = metricsFromProgram.drop(['', 'Functions', 'Line of Code', 'Mutants', 'Minimals', '%', 'Equivalents', 'Test Cases'], axis=1)
	
	# Renaming columns by removing _F1 at the end
	metricsFromProgram = metricsFromProgram.rename(lambda x: x.replace('_F1', '') ,axis='columns')

	# Each column_classifier is a value in the dictionary (Example: EM_KNN, MM_RF, etc ...)
	metricsFromClassifier = pd.DataFrame()

	for program, values in metricsFromProgram.iterrows():
		programDataFrame = pd.DataFrame() #Dataframe containing all the classifiers for the program and due F1 for this classifiers
		for key, value in values.iteritems():
			classifier = key[key.index('_') + 1 : ]
			column = 'EQUIVALENT' if key[ : key.index('_')] == 'EM' else 'MINIMAL'
			f1 = float(value)
			
			newDataFrame = pd.DataFrame(data=[[ classifier, column, program, f1 ]], columns=['Classifier', 'Column', 'Program', 'F1'])
			programDataFrame = programDataFrame.append(newDataFrame)
			metricsFromClassifier = metricsFromClassifier.append(newDataFrame)

	if plot:
		plotClassifiersProgramAProgram(metricsFromClassifier, possibleClassifiers)

	return metricsFromClassifier

def plotClassifiersProgramAProgram(metricsFromClassifier, possibleClassifiers, mean_median = 0):
	'''
		Function that generates a box plot graph with the F1 score of the classifiers for each program.
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
	ax.set_xlabel('\nClassifiers', fontsize = 22)
	ax.set_ylabel('F1 Score', fontsize = 22)

	#Box width
	boxWidth = 0.7

	# Set the boxplot positions
	positionsMM = [value for value in range(len(possibleClassifiers) * 2) if value % 2 == 0 ]
	positionsEM = [value + boxWidth for value in range(len(possibleClassifiers) * 2) if value % 2 == 0 ]

	# Set the boxplots based on a function property defined below
	bpMM = boxPlotProperties(ax, dataMinimal, positionsMM, '#607D8B', boxWidth)
	bpEM = boxPlotProperties(ax, dataEquivalent, positionsEM, '#B0BEC5', boxWidth)

	# 0 - Mean | 1 - Median
	# Calculate the means / medians
	if mean_median == 0:
		minimalMean_Medians = [ np.round(np.mean(value), 2) for value in dataMinimal]
		equivalentMean_Medians = [np.round(np.mean(value), 2) for value in dataEquivalent]
	else:
		minimalMean_Medians = [ np.round(np.median(value), 2) for value in dataMinimal]
		equivalentMean_Medians = [np.round(np.median(value), 2) for value in dataEquivalent]

	# Display values in boxplot
	for tick in range(len(positionsMM)):
		# Display a median on boxplot top
		ax.text(positionsMM[tick], 101, minimalMean_Medians[tick], horizontalalignment='center', size=16, color='black')
		ax.text(positionsEM[tick], 101, equivalentMean_Medians[tick], horizontalalignment='center', size=16, color='black')

        # Display a median on boxplot
		#ax.text(positionsMM[tick], minimalMeans[tick] + 0.5, minimalMeans[tick], horizontalalignment='center', size='small', color='black', weight='semibold')
		#ax.text(positionsEM[tick], equivalentMeans[tick] + 0.5, equivalentMeans[tick], horizontalalignment='center', size='small', color='black', weight='semibold')

	# Set the label between two boxplot
	#ax.set_xticklabels(list(getFullNamePossibleClassifiers().values())) # Caso for exibir o nome completo do classificador
	ax.set_xticklabels(possibleClassifiers)
	for tick in ax.xaxis.get_major_ticks(): tick.label.set_fontsize(18)
	for tick in ax.yaxis.get_major_ticks(): tick.label.set_fontsize(16)
	ax.set_xticks([value + (boxWidth / 2) for value in ax.get_xticks() if value % 2 == 0])

	# Set the chart subtitle/legend
	ax.legend([bpMM["boxes"][0], bpEM["boxes"][0]], ['Minimal Mutants', 'Equivalent Mutants'], loc='lower right', fontsize='xx-large')

	fig.tight_layout()

	# Display chart
	plt.show()

def boxPlotProperties(ax, data, positions, color, boxWidth):
	boxprops = dict(facecolor=color, linewidth = 1)
	medianprops = dict(color='#000000', linewidth = 1)
	meanprops = dict(color='#000000', linewidth = 1)

	boxplot = ax.boxplot(data, sym='+', positions=positions, patch_artist=True, meanline=True,
		showmeans=True, labels=possibleClassifiers, boxprops=boxprops, 
		medianprops=medianprops, meanprops=meanprops, widths=boxWidth)

	return boxplot

def analyzeClassificationsFromEachProgram(targetColumn, possiblePrograms, overwrite = False):
	'''
		Função responsável por fazer a análise dos resultados das classificações dos mutantes dos programas e obter as métricas para cada programa
	'''	
	baseFolder = '{}/ML/Results/{}/Classification'.format(os.getcwd(), targetColumn)
	fileName = 'ML_Metrics'
	metricsFile = '{}/ML/Results/{}/Classification/{}.csv'.format(os.getcwd(), targetColumn, fileName)

	mutantsMetrics = pd.DataFrame()

	if overwrite or not util.pathExists(metricsFile):

		for file in util.getFilesInFolder(baseFolder):
			programName = util.getPathName(file)
			programName = programName[ : programName.find('.')]
			
			if programName in possiblePrograms:
				classificationResult = util.createDataFrameFromCSV(file, hasHeader = True)
				
				y_correct = classificationResult.loc[ : , '_IM_{}'.format(targetColumn)].values
				y_predicted = classificationResult.loc[ : , 'RESULT'].values

				accuracy, precision, recall, f1, TPR, FPR, TP, FN, FP, TN = ML_Mutants.evaluatingClassification(y_correct, y_predicted)

				newMutantsMetrics = pd.DataFrame(data=[[programName, accuracy, precision, recall, f1]], columns=['ProgramName', 'Accuracy', 'Precision', 'Recall', 'F1'])
				mutantsMetrics = mutantsMetrics.append(newMutantsMetrics)

				#print('Program: {}\t\tAccuracy: {} | Precision: {} | Recall: {} | F1: {}'.format(programName, accuracy, precision, recall, f1))

		util.writeDataFrameInCsvFile(metricsFile, mutantsMetrics)

		return mutantsMetrics
	elif util.pathExists(metricsFile):
		return util.createDataFrameFromCSV(metricsFile, True)	

def summarizeClassifications(targetColumn, possiblePrograms, df_Programs_BestClassifiers, overwrite = False):
	'''
		Função responsável por fazer a análise dos resultados das classificações dos mutantes dos programas e sumarizar todos os resultados.
	'''
	baseFolder = '{}/ML/Results/{}/Classification'.format(os.getcwd(), targetColumn)
	fileName = 'ClassificationSummary'
	summaryFile = '{}/ML/Results/{}/Classification/{}.csv'.format(os.getcwd(), targetColumn, fileName)

	mutantsData = pd.DataFrame()

	df_Programs_BestClassifiers = df_Programs_BestClassifiers.query('Column == \'{}\''.format(targetColumn))

	if overwrite or not util.pathExists(summaryFile):

		for programName in possiblePrograms:
			bestClassifier = df_Programs_BestClassifiers.query('ProgramName == \'{}\''.format(programName)).loc[ : , 'Classifier'].values[0]
			programFileBestClassifier = '{}/{}_{}.csv'.format(baseFolder, programName, bestClassifier)

			if util.pathExists(programFileBestClassifier):
				classificationResult = util.createDataFrameFromCSV(programFileBestClassifier, hasHeader=True)

				mutantsData = mutantsData.append(classificationResult, ignore_index = True)

		util.writeDataFrameInCsvFile(summaryFile, mutantsData)

		return mutantsData
	elif util.pathExists(summaryFile):
		return util.createDataFrameFromCSV(summaryFile, True)

def plotSummarizedClassifications(mutantsData, column):
	plotSummarizedClassifications_byColumn(mutantsData, column, '_IM_OPERATOR', calculateAboveMedian = True, show = False, plot = True)
	#plotSummarizedClassifications_byColumn(mutantsData, column, '_IM_TYPE_STATEMENT', show = True)

def plotSummarizedClassifications_byColumn(mutantsData, targetColumn, columnToAnalyze, show = False, calculateAboveMedian = False, plot = False):
	# --- Analyze for operators
	columnData = pd.DataFrame(mutantsData).loc[ : , [targetColumn, '_IM_PROGRAM', columnToAnalyze, 'PREDICTED', 'RESULT']]

	columnInfos = pd.DataFrame()
	
	distinctInfoColumn = columnData[columnToAnalyze].unique()
	mutantsNumber = [ columnData.query('{} == \'{}\''.format(columnToAnalyze, operator))['RESULT'].count() for operator in distinctInfoColumn] # Número de mutantes/linhas
	
	value_1 = [ columnData.query('{} == \'{}\' and {} == \'1\''.format(columnToAnalyze, operator, targetColumn))['RESULT'].count() for operator in distinctInfoColumn] # Mutantes classificados como 1
	value_0 = [ columnData.query('{} == \'{}\' and {} == \'0\''.format(columnToAnalyze, operator, targetColumn))['RESULT'].count() for operator in distinctInfoColumn] # Mutantes classificados como 0
	
	correctPredictions = [ columnData.query('{} == \'{}\' and RESULT == \'1\''.format(columnToAnalyze, operator))['RESULT'].count() for operator in distinctInfoColumn] # Predições corretas
	percCorrectPredictions = [ value * 100 / mutantsNumber[iCount] for iCount, value in zip(range(len(correctPredictions)), correctPredictions) ] # % Predições corretas
	
	wrongPredictions = [ columnData.query('{} == \'{}\' and RESULT == \'0\''.format(columnToAnalyze, operator))['RESULT'].count() for operator in distinctInfoColumn] # Predições incorretas
	percWrongPredictions = [ value * 100 / mutantsNumber[iCount] for iCount, value in zip(range(len(wrongPredictions)), wrongPredictions) ] # % Predições incorretas
	
	correctPredictionsFor1 = [ columnData.query('{} == \'{}\' and PREDICTED == \'1\' and RESULT == \'1\''.format(columnToAnalyze, operator))['RESULT'].count() for operator in distinctInfoColumn] # Predições corretas para 1
	percCorrectPredictionsFor1 = [ value * 100 / value_1[iCount] for iCount, value in zip(range(len(correctPredictionsFor1)), correctPredictionsFor1) ] # % Predições corretas para 1
	
	correctPredictionsFor0 = [ columnData.query('{} == \'{}\' and PREDICTED == \'0\' and RESULT == \'1\''.format(columnToAnalyze, operator))['RESULT'].count() for operator in distinctInfoColumn] # Predições corretas para 0
	percCorrectPredictionsFor0 = [ value * 100 / value_0[iCount] for iCount, value in zip(range(len(correctPredictionsFor0)), correctPredictionsFor0) ] # % Predições corretas para 0
	
	wrongPredictionsFor1 = [ columnData.query('{} == \'{}\' and PREDICTED == \'1\' and RESULT == \'0\''.format(columnToAnalyze, operator))['RESULT'].count() for operator in distinctInfoColumn] # Predições para 1 sendo que era 0
	percWrongPredictionsFor1 = [ value * 100 / value_1[iCount] for iCount, value in zip(range(len(wrongPredictionsFor1)), wrongPredictionsFor1) ] # % Predições para 1 sendo que era 0
	
	wrongPredictionsFor0 = [ columnData.query('{} == \'{}\' and PREDICTED == \'0\' and RESULT == \'0\''.format(columnToAnalyze, operator))['RESULT'].count() for operator in distinctInfoColumn] # Predições para 0 sendo que era 1
	percWrongPredictionsFor0 = [ value * 100 / value_0[iCount] for iCount, value in zip(range(len(wrongPredictionsFor0)), wrongPredictionsFor0) ] # % Predições para 1 sendo que era 0

	columns = [columnToAnalyze, 'Number', 'Value_1', 'Value_0', 'Correct', 'Correct_Perc', 'Wrong', 'Wrong_Perc', 'Correct_1', 'Correct_1_Perc', 'Correct_0', 'Correct_0_Perc', 'Wrong_1', 'Wrong_1_Perc', 'Wrong_0', 'Wrong_0_Perc']
	data = np.array([distinctInfoColumn, mutantsNumber, value_1, value_0, correctPredictions, percCorrectPredictions, wrongPredictions, percWrongPredictions, correctPredictionsFor1, percCorrectPredictionsFor1, correctPredictionsFor0, percCorrectPredictionsFor0, wrongPredictionsFor1, percWrongPredictionsFor1, wrongPredictionsFor0, percWrongPredictionsFor0]).transpose()

	columnInfos = pd.DataFrame(data=data, columns=columns)

	# Get the median to rank only operators that are at least above the median
	if calculateAboveMedian:
		median = np.median(columnInfos['Number'])
	else:
		median = 0

	# Operadores com melhor classificação geral (apenas com número de mutantes acima do 3º quartil)
	op_best = columnInfos.query('Number >= {}'.format(median)).sort_values('Correct_Perc', ascending=False)
	if show: print('Melhores ' + columnToAnalyze + ' na classificação geral\n', op_best.head(n = 10), end='\n\n')
	
	# Operadores com pior classificação geral
	op_poor = columnInfos.query('Number >= {}'.format(median)).sort_values('Correct_Perc', ascending=True)
	if show: print('Piores ' + columnToAnalyze + ' na classificação geral\n', op_poor.head(n = 10), end='\n\n')
	
	# Operadores com melhor apontamento dos minimais/equivalentes
	op_best_value1 = columnInfos.query('Number >= {}'.format(median)).sort_values('Correct_1_Perc', ascending=False)
	if show: print('Melhores ' + columnToAnalyze + ' ao apontar os minimais/equivalentes\n', op_best_value1.head(n = 10), end='\n\n')
	
	# Operadores com melhor apontamento dos não minimais/equivalentes
	op_best_value0 = columnInfos.query('Number >= {}'.format(median)).sort_values('Correct_0_Perc', ascending=False)
	if show: print('Melhores ' + columnToAnalyze + ' ao apontar os não minimais/equivalentes\n', op_best_value0.head(n = 10), end='\n\n')
	
	# Operadores com pior apontamento dos minimais/equivalentes
	op_poor_value1 = columnInfos.query('Number >= {}'.format(median)).sort_values('Correct_1_Perc', ascending=True)
	if show: print('Piores ' + columnToAnalyze + ' ao apontar os minimais/equivalentes\n', op_poor_value1.head(n = 10), end='\n\n')
	
	# Operadores com pior apontamento dos não minimais/equivalentes
	op_poor_value0 = columnInfos.query('Number >= {}'.format(median)).sort_values('Correct_0_Perc', ascending=True)
	if show: print('Piores ' + columnToAnalyze + ' ao apontar os não minimais/equivalentes\n', op_poor_value0.head(n = 10), end='\n\n')

	if plot:
		plotData = op_best.loc[ : , [columnToAnalyze, 'Correct_Perc'] ].head()
		plotData = plotData.append(op_best.loc[ : , [columnToAnalyze, 'Correct_Perc'] ].tail())
		
		# Create the figure with axis
		fig = plt.figure(1, figsize=(9, 6))
		ax = fig.add_subplot(1, 1, 1)
		
		# Set the value to be shown as indexes on axis Y
		#ax.set_yticks([value for value in range(0, 101, 10)])
		ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.75)
		
		# Set the chart title and axis title
		#ax.set_title('axes title', fontsize = 14)
		ax.set_xlabel('\n' + columnToAnalyze, fontsize = 14)
		ax.set_ylabel('Number of programs that the classifier was the best', fontsize = 14)
		
		# Set the boxplot positions
		width = 0.4  # the width of the bars
		#positionsMM = [value - width / 2 for value in range(len(possibleClassifiers))]
		positionsMM = [value for value in range(len(plotData))]
		
		# Set the bar chart based on a function property defined below
		bcMM = barChartProperties(ax, plotData.loc[ : , 'Correct_Perc' ].values, positionsMM, '#5975a4', width)
		
		# Set the label between two boxplot
		ax.set_xticklabels(plotData.loc[ : , columnToAnalyze ].values)
		ax.set_xticks([value for value in range(len(plotData))])
		
		# Set the chart subtitle/legend
		#ax.legend([bcMM, bcEM], ['Minimal Mutants', 'Equivalent Mutants'], loc='upper left')
		#ax.legend(loc='upper left')
		
		autolabel(bcMM, ax, decimals=2)
		
		fig.tight_layout()
		
		# Display chart
		plt.show()

def getBestClassifierForPrograms(program = None, targetColumn = None, write = True, overwrite = False):
	'''
		Function responsible to analyze the results from all programs with all classifiers
		If the parameter 'writeFile' is True, this function writes a file with the result
		Returns a dataframe with the best classifier for each program and this metrics
	'''
	
	# SetUp
	possibleTargetColumns, possibleClassifiers, possiblePrograms = setUp()

	# Indicate the best classifier for each classifier - ProgramName, Column, Classifier, Accuracy, Precision, Recall, F1
	df_Programs_BestClassifiers = pd.DataFrame()

	# Set possible none parameters
	if program is not None:
		programs = [program]
	else:
		programs = possiblePrograms

	# Set possible none parameters
	if targetColumn is not None:
		columns = [targetColumn]
	else:
		columns = possibleTargetColumns

	mustWrite = write and program is None and targetColumn is None

	if mustWrite:
		for _column in columns:
			baseFolder = '{}/ML/Results/{}/Classification'.format(os.getcwd(), _column)

			# Dataframe containing programs, the best classifier for each one and the metrics achieved
			df_Program_Metrics_BestClassifier = pd.DataFrame()

			# Dataframe containing programs and all classifiers for each one and the metrics achieved
			df_Program_Metrics_AllClassifiers = pd.DataFrame()

			for programName in programs:
				df_Program_Classifiers_Metrics = pd.DataFrame()
				
				for classifier in possibleClassifiers:
					fileName = '{}/{}_{}.csv'.format(baseFolder, programName, classifier)

					if util.pathExists(fileName):
						classificationResult = util.createDataFrameFromCSV(fileName, hasHeader = True)

						y_correct = classificationResult.loc[ : , '_IM_{}'.format(_column)].values
						y_predicted = classificationResult.loc[ : , 'PREDICTED'].values

						#accuracy, precision, recall, f1, TPR, FPR, TP, FN, FP, TN = ML_Mutants.evaluatingClassification(y_correct, y_predicted)		
						accuracy, precision, recall, f1, _, _, _, _, _, _ = ML_Mutants.evaluatingClassification(y_correct, y_predicted)

						#print('Program: {} | Column: {} | Classifier: {} | Accuracy: {} | Precision: {} | Recall: {} | F1: {}'.format(programName, _column, classifier, accuracy, precision, recall, f1))
						newDataFrame = pd.DataFrame(data=[[programName, _column, classifier, accuracy * 100, precision * 100, recall * 100, f1 * 100]], columns=['ProgramName', 'Column', 'Classifier', 'Accuracy', 'Precision', 'Recall', 'F1'])
						df_Program_Classifiers_Metrics = df_Program_Classifiers_Metrics.append(newDataFrame)

				df_Program_Metrics_AllClassifiers = df_Program_Metrics_AllClassifiers.append(df_Program_Classifiers_Metrics.sort_values('F1', ascending=False))
				
				# Classify here the best classifier for these program
				df_Program_Classifiers_Metrics = df_Program_Classifiers_Metrics.sort_values('F1', ascending=False).head(n=1)
				df_Program_Metrics_BestClassifier = df_Program_Metrics_BestClassifier.append(df_Program_Classifiers_Metrics)
				df_Programs_BestClassifiers = df_Programs_BestClassifiers.append(df_Program_Classifiers_Metrics)

			# Save the dataframe into file if is specified to read all program files
			if mustWrite:
				fileName = '{}/Metrics.csv'.format(baseFolder)
				fileNameAllClassifiers = '{}/Metrics_AllClassifiers.csv'.format(baseFolder)
				if not util.pathExists(fileName) or overwrite:
					util.writeDataFrameInCsvFile(fileName, df_Program_Metrics_BestClassifier)
					util.writeDataFrameInCsvFile(fileNameAllClassifiers, df_Program_Metrics_AllClassifiers)
	else:
		for _column in columns:
			baseFolder = '{}/ML/Results/{}/Classification'.format(os.getcwd(), _column)
			fileName = '{}/Metrics.csv'.format(baseFolder)

			df_Programs_BestClassifiers = df_Programs_BestClassifiers.append(util.createDataFrameFromCSV(fileName, True))

	return df_Programs_BestClassifiers

def plotMetricsFromBestClassifiersOfEachProgram(df_Programs_BestClassifiers):
	targetColumnsOption = {'MINIMAL': ('#607D8B', '#B0BEC5', 'Minimal Mutants'), 'EQUIVALENT': ('#607D8B', '#B0BEC5', 'Equivalent Mutants')}

	for targetColumn, (colorF1, colorAccuracy, columnName) in targetColumnsOption.items():
		targetColumnsF1 = []
		targetColumnsAccuracy = []

		# Filter by column
		df_Programs_BestClassifiers_Column = df_Programs_BestClassifiers.query('Column == \'{}\''.format(targetColumn))

		# Rename the ProgramName column to upper case, sort the df ordering by ProgramName, insert the ProgramID in the DF and resort the df by F1 Value
		df_Programs_BestClassifiers_Column = df_Programs_BestClassifiers_Column.replace(to_replace=df_Programs_BestClassifiers_Column['ProgramName'].unique(), value=[program.upper() for program in df_Programs_BestClassifiers_Column['ProgramName'].unique()]).sort_values('ProgramName')
		df_Programs_BestClassifiers_Column.insert(0, 'ProgramID', ['#{}'.format(value + 1) for value in range(len(df_Programs_BestClassifiers_Column['ProgramName'].unique()))])
		df_Programs_BestClassifiers_Column = df_Programs_BestClassifiers_Column.sort_values('F1', ascending=False)
		
		programsName = df_Programs_BestClassifiers_Column['ProgramName'].unique()
		programsID = df_Programs_BestClassifiers_Column['ProgramID'].unique()

		for programName in programsName:
			F1Value = df_Programs_BestClassifiers_Column.query('ProgramName == \'{}\''.format(programName))['F1'].values[0]
			AccuracyValue = df_Programs_BestClassifiers_Column.query('ProgramName == \'{}\''.format(programName))['Accuracy'].values[0]

			targetColumnsF1.append(F1Value)
			targetColumnsAccuracy.append(AccuracyValue)

		# Create the figure with axis
		fig = plt.figure(1, figsize=(9, 6))
		ax = fig.add_subplot(1, 1, 1)

		# Set the value to be shown as indexes on axis Y
		ax.set_yticks([value for value in range(0, 50, 10)] + [value for value in range(50, 101, 5)])
		ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.75)

		# Set the chart title and axis title
		#ax.set_title('Metrics for each program with the best classifier - {}'.format(columnName), fontsize = 22)
		ax.set_xlabel('Programs', fontsize = 16)
		ax.set_ylabel('F1 Score and Accuracy', fontsize = 16)

		# Set the boxplot positions
		#width = 0.2  # the width of the bars
		positions = [value for value in range(len(programsName))]

		# Set the bar chart based on a function property defined below
		barChartAccuracy = barChartProperties(ax, targetColumnsAccuracy, positions, colorAccuracy, None)
		barChartF1 = barChartProperties(ax, targetColumnsF1, positions, colorF1, None)

		# Set the label between two boxplot
		ax.set_xticklabels([programID for programID in programsID])
		ax.set_xticks([value for value in range(len(programsName))])

		# Set the chart subtitle/legend
		ax.legend([barChartF1, barChartAccuracy], ['F1 Score', 'Accuracy'], loc='upper center', fontsize='xx-large')

		autolabel(barChartAccuracy, ax, 2)
		autolabel(barChartF1, ax, 2)

		fig.tight_layout()

		# Display chart
		plt.show()


def setUp():
	possibleTargetColumns = ['MINIMAL', 'EQUIVALENT']
	possibleClassifiers = getPossibleClassifiers()
	possiblePrograms = [ util.getPathName(program) for program in util.getPrograms()]

	return possibleTargetColumns, possibleClassifiers, possiblePrograms

if __name__ == '__main__':
	# ---------
	# --- Setup
	possibleTargetColumns, possibleClassifiers, possiblePrograms = setUp()

	# ---------------------------------------------------------------------------------------------------
	# --- Analyze the 30 runs and calc statistics informations, like minimum, maximum, median and average
	analyzeResults(possibleTargetColumns, possibleClassifiers, plot = True)
	
	# ----------------------------------
	# --- Get informations from programs
	#programsInfo = getProgramsInfo()
	#programsInfo = getMetricsFromPrograms(possibleTargetColumns, possibleClassifiers, programsInfo, bestParameter=True)
	
	#programsBestMetrics = analyzeMetricsFromProgram(programsInfo, possibleClassifiers, plot=False)
	
	#metricsFromClassifier = analyzeClassifiersProgramAProgram(programsInfo, possibleClassifiers, plot=True)
	
	#df_Programs_BestClassifiers = getBestClassifierForPrograms()
	#plotMetricsFromBestClassifiersOfEachProgram(df_Programs_BestClassifiers)

	# ---------------------------------------------
	# --- File analysis with classified mutant data
	#minimalMutantsMetrics = analyzeClassificationsFromEachProgram(possibleTargetColumns[0], possiblePrograms, True)
	#equivalentMutantsMetrics = analyzeClassificationsFromEachProgram(possibleTargetColumns[1], possiblePrograms, True)

	# --------------------------------------------------------
	# --- File analysis with summarized classified mutant data
	#minimalsMutantsData = summarizeClassifications(possibleTargetColumns[0], possiblePrograms, df_Programs_BestClassifiers, True)
	#plotSummarizedClassifications(minimalsMutantsData, '_IM_' + possibleTargetColumns[0])
	
	#equivalentsMutantsData = summarizeClassifications(possibleTargetColumns[1], possiblePrograms, df_Programs_BestClassifiers, True)
	#plotSummarizedClassifications(equivalentsMutantsData, '_IM_' + possibleTargetColumns[1])