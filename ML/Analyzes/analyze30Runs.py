# --------------
# --- Statistics
from statistics import mean
from statistics import median

# ------
# --- OS
import os

# -------
# --- sys
import sys

# ----------
# --- Pandas
import pandas as pd

# ---------
# --- NumPy
import numpy as np

# ---------------
# --- Matplot Lib
import matplotlib.pyplot as plt

# ------------
# --- Analyzes
from analyzesUtil import barChartProperties, autolabel, boxPlotProperties

def getRunsOnlyFromBestParameter(experimentResults, classifiersBestParameter, possibleTargetColumns):
	for parametrizedClassifier in ['KNN', 'DT', 'RF']:
		for targetColumn in possibleTargetColumns:
			bestParameter = classifiersBestParameter.query('TargetColumn == \'{}\' and Classifier == \'{}\''.format(targetColumn, parametrizedClassifier))['Parameter'].values[0]
			toBeDeleted = experimentResults.query('TargetColumn == \'{}\' and Classifier == \'{}\' and SampleSplit != \'{}\''.format(targetColumn, parametrizedClassifier, bestParameter))
			indexesToBeDeleted = toBeDeleted.index.values

			experimentResults = experimentResults.drop(labels = indexesToBeDeleted, axis = 0)

	return experimentResults

def summarizeRunsFromCustomParameter(experimentResults):
	"""
	Função que faz a análise de todas as execuções de toda
	"""
	possibleClassifiers = experimentResults['Classifier'].unique()
	possibleTargetColumn = experimentResults['TargetColumn'].unique()

	customParameterSummaryResults = pd.DataFrame()

	for classifier in possibleClassifiers:
		for targetColumn in possibleTargetColumn:
			classifierColumnResults = experimentResults.query('TargetColumn == \'{}\' and Classifier == \'{}\''.format(targetColumn, classifier))

			for sampleSplit in classifierColumnResults['SampleSplit'].unique():
				value = classifierColumnResults.query('SampleSplit == \'{}\''.format(sampleSplit))

				accuracy = np.mean(value['Accuracy'])
				precision = np.mean(value['Precision'])
				recall = np.mean(value['Recall'])
				f1 = np.mean(value['F1'])

				#parameterMetrics = parameterMetrics.append(pd.DataFrame(data=[[targetColumn, classifier, parameter, meanAccuracy, meanPrecision, meanRecall, meanF1]], columns=['TargetColumn', 'Classifier', 'Parameter', 'Accuracy', 'Precision', 'Recall', 'F1']))
				customParameterSummaryResults = customParameterSummaryResults.append(pd.DataFrame(data=[[targetColumn, classifier, sampleSplit, accuracy, precision, recall, f1]], columns=['TargetColumn', 'Classifier', 'SampleSplit', 'Accuracy', 'Precision', 'Recall', 'F1']))

	return customParameterSummaryResults


def getRunsFromCustomParameters(experimentResults):
	toBeDeleted = experimentResults.query('SampleSplit == 0')
	indexesToBeDeleted = toBeDeleted.index.values
	experimentResults = experimentResults.drop(labels = indexesToBeDeleted, axis = 0)

	return experimentResults

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

				targetColumnData[possibleClassifiers.index(classifier)] = Values_Classifier_Column['F1'].values[0]

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
		for _ in range(len(possibleClassifiers)):
			dataMinimal.append(0)
			dataEquivalent.append(0)

		for column in possibleTargetColumns:
			for classifier in possibleClassifiers:
				Values_Classifier_Column = runsResults.query('Classifier == \'{}\' and TargetColumn == \'{}\' '.format(classifier, column))

				#print('Classifier: {}\tColumn: {}\t\tAccuracy: {:.2f}\tPrecision: {:.2f}\tRecall: {:.2f}\tF1: {:.2f}'.format(classifier, column, meanAccuracy, meanPrecision, meanRecall, meanF1))

				if column == 'MINIMAL':
					dataMinimal[possibleClassifiers.index(classifier)] = Values_Classifier_Column['F1'].values[0]
				elif column == 'EQUIVALENT':
					dataEquivalent[possibleClassifiers.index(classifier)] = Values_Classifier_Column['F1'].values[0]

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

def plotRunsDetailed(runsResults, possibleClassifiers, possibleTargetColumns, mean_median = 0):
	# Set lists with data. For each array, there are several internal arrays containing the F1 values for each program
	dataMinimal = []
	dataEquivalent = []
	for classifier in possibleClassifiers:
		dataMinimal.append([])
		dataEquivalent.append([])
	
	for column in possibleTargetColumns:
		for classifier in possibleClassifiers:
			Values_Classifier_Column = runsResults.query('Classifier == \'{}\' and TargetColumn == \'{}\' '.format(classifier, column))

			#print('Classifier: {}\tColumn: {}\t\tAccuracy: {:.2f}\tPrecision: {:.2f}\tRecall: {:.2f}\tF1: {:.2f}'.format(classifier, column, meanAccuracy, meanPrecision, meanRecall, meanF1))

			if column == 'MINIMAL':
				dataMinimal[possibleClassifiers.index(classifier)] = Values_Classifier_Column['F1'].values
			elif column == 'EQUIVALENT':
				dataEquivalent[possibleClassifiers.index(classifier)] = Values_Classifier_Column['F1'].values
	
	# Create the figure with axis
	fig = plt.figure(1, figsize=(9, 6))
	ax = fig.add_subplot(1, 1, 1)

	# Set the value to be shown as indexes on axis Y
	#ax.set_yticks([value for value in range(0, 101, 10)])
	ax.set_yticks([value for value in range(0, 60, 10)] + [value for value in range(60, 101, 5)])
	ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.75)

	# Set the chart title and axis title
	#ax.set_title('Resultados do F1-Score das 30 execuções para cada classificador\n\n', fontsize = 26) #Portuguese
	#ax.set_xlabel('\nClassifiers', fontsize = 22) #English
	ax.set_xlabel('\nClassificadores', fontsize = 22) #Portuguese
	ax.set_ylabel('F1 Score', fontsize = 22)

	#Box width
	boxWidth = 0.7

	# Set the boxplot positions
	positionsMM = [value for value in range(len(possibleClassifiers) * 2) if value % 2 == 0 ]
	positionsEM = [value + boxWidth for value in range(len(possibleClassifiers) * 2) if value % 2 == 0 ]

	# Set the boxplots based on a function property defined below
	bpMM = boxPlotProperties(ax, dataMinimal, positionsMM, '#607D8B', boxWidth, possibleClassifiers)
	bpEM = boxPlotProperties(ax, dataEquivalent, positionsEM, '#B0BEC5', boxWidth, possibleClassifiers)

	# Calculate the standard deviation
	minimalMean_Medians = [ np.round(np.std(value), 2) for value in dataMinimal]
	equivalentMean_Medians = [np.round(np.std(value), 2) for value in dataEquivalent]

	# Display values in boxplot
	for tick in range(len(positionsMM)):
		# Display a info on boxplot top
		_, topBoxplot = ax.get_ylim()
		ax.text(positionsMM[tick], topBoxplot + 1, minimalMean_Medians[tick], horizontalalignment='center', size=16, color='black')
		ax.text(positionsEM[tick], topBoxplot + 1, equivalentMean_Medians[tick], horizontalalignment='center', size=16, color='black')

	# Set the label between two boxplot
	#ax.set_xticklabels(list(getFullNamePossibleClassifiers().values())) # Caso for exibir o nome completo do classificador
	ax.set_xticklabels(possibleClassifiers)
	for tick in ax.xaxis.get_major_ticks(): tick.label.set_fontsize(18)
	for tick in ax.yaxis.get_major_ticks(): tick.label.set_fontsize(16)
	ax.set_xticks([value + (boxWidth / 2) for value in ax.get_xticks() if value % 2 == 0])

	# Set the chart subtitle/legend
	#ax.legend([bpMM["boxes"][0], bpEM["boxes"][0]], ['Minimal Mutants', 'Equivalent Mutants'], loc='lower left', fontsize='xx-large') #English
	ax.legend([bpMM["boxes"][0], bpEM["boxes"][0]], ['Mutantes Minimais', 'Mutantes Equivalentes'], loc='lower left', fontsize='xx-large') #Portuguese

	fig.tight_layout()

	# Display chart
	plt.show()