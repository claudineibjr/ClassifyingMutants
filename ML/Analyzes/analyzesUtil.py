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

# -----------
# --- SkLearn
from sklearn.metrics import confusion_matrix

# --------
# --- Util
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
super_parent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, super_parent_dir)
import util

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

def barChartProperties(ax, data, positions, color, width):
	if width is None:
		barChart = ax.bar(positions, data, color=color)
	else:
		barChart = ax.bar(positions, data, width=width, color=color)
	
	return barChart

def getProgramsHeader():
	programsInfoFileName = '{}/Programs/ProgramsInfo.csv'.format(os.getcwd())
	programsInfo = util.getContentFromFile(programsInfoFileName)

	columns = [line.split(',') for line in programsInfo.splitlines()][ : 1]
	
	return columns[0]

def bestParameterFileExists(file):
	fileFilter = '_bestParameter'
	if file.__contains__(fileFilter):
		if util.pathExists(file):
			return file
		else:
			return file.replace(fileFilter, '')
	else:
		return file

def boxPlotProperties(ax, data, positions, color, boxWidth, possibleClassifiers):
	boxprops = dict(facecolor=color, linewidth = 1)
	medianprops = dict(color='#000000', linewidth = 1)
	meanprops = dict(color='#000000', linewidth = 1)

	boxplot = ax.boxplot(data, sym='+', positions=positions, patch_artist=True, meanline=True,
		showmeans=True, labels=possibleClassifiers, boxprops=boxprops, 
		medianprops=medianprops, meanprops=meanprops, widths=boxWidth)

	return boxplot

def evaluatingClassification(y_test, y_pred):
	##################################
	# --- Evaluating the Algorithm ---
	##################################
	#   For evaluating an algorithm, confusion matrix, precision, recall and f1 score are the most commonly used metrics. The confusion_matrix and classification_report methods of the sklearn.metrics can be used to calculate these metrics. Take a look at the following script:
	confusionMatrix = confusion_matrix(y_test, y_pred)
	# =COUNTIFS(P2:P50000, 1, O2:O50000, 1)
	TP = confusionMatrix[1][1]  # True Positives	
	# =COUNTIFS(P2:P50000, 0, O2:O50000, 1)
	FN = confusionMatrix[1][0]  # False Negatives	
	# =COUNTIFS(P2:P50000, 1, O2:O50000, 0)
	FP = confusionMatrix[0][1]  # False Positives	
	# =COUNTIFS(P2:P50000, 0, O2:O50000, 0)
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
	# =((S2 + V2) / (U2 + T2 + S2 + V2)) * 100
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
	# =((S2) / (S2 + U2)) * 100
	precision = (TP) / (TP + FP)
	# =((S2) / (T2 + S2)) * 100
	recall = (TP) / (FN + TP)
	
	########
	# --- F1
	#   In practice, often a combination of precision and recall is used, the so-called F1-score.
	# =2 * ((X2 * Y2) / (X2 + Y2))
	f1 = 2 * ((precision * recall) / (precision + recall))

	return accuracy, precision, recall, f1, TPR, FPR, TP, FN, FP, TN