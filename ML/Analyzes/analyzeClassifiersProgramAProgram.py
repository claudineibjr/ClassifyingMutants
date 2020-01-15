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

# --------
# --- Util
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
super_parent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, super_parent_dir)
import util

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
	bpMM = boxPlotProperties(ax, dataMinimal, positionsMM, '#607D8B', boxWidth, possibleClassifiers)
	bpEM = boxPlotProperties(ax, dataEquivalent, positionsEM, '#B0BEC5', boxWidth, possibleClassifiers)

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