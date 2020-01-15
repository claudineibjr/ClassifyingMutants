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

# ------------
# --- Analyzes
from analyzesUtil import barChartProperties, autolabel, boxPlotProperties

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
		for _ in range(len(possibleClassifiers)):
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