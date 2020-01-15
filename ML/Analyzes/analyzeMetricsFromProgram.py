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
from analyzesUtil import barChartProperties, autolabel

# --------
# --- Util
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
super_parent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, super_parent_dir)
import util

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