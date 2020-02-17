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
from analyzesUtil import evaluatingClassification

# --------
# --- Util
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
super_parent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, super_parent_dir)
import util

from shutil import copyfile

def getMLMetricsFromClassificationFile(fileName, targetColumn, programName):
	# Obtém as métricas do programa selecionado
	classificationResult = util.createDataFrameFromCSV(fileName, hasHeader = True)
	
	y_correct = classificationResult.loc[ : , '_IM_{}'.format(targetColumn)].values
	y_predicted = classificationResult.loc[ : , 'PREDICTED'].values

	accuracy, precision, recall, f1, TPR, FPR, TP, FN, FP, TN = evaluatingClassification(y_correct, y_predicted)

	#newMutantsMetrics = pd.DataFrame(data=[[programName, accuracy, precision, recall, f1]], columns=['ProgramName', 'Accuracy', 'Precision', 'Recall', 'F1'])	
	
	return (accuracy, precision, recall, f1)


def getBestClassifierForEachProgram(possibleTargetColumns, possiblePrograms, possibleClassifiers, bestProgram_Classifier_Parameters, writeFile = False):
	bestProgram_Classifier = pd.DataFrame()

	for targetColumn in possibleTargetColumns:
		baseFolder = '{}/ML/Results/{}/Classification'.format(os.getcwd(), targetColumn)
		fileName = '{}/bestClassifierForEachProgram.csv'.format(baseFolder)

		if writeFile:

			# Percorre todos os programas
			for programName in possiblePrograms:
				program_ClassifierMetrics = pd.DataFrame()
				#print('Column: {} | Program: {}'.format(targetColumn, programName))

				# Percorre todos os classificadores daquele programa		
				for classifier in possibleClassifiers:
					programFileName = '{}/{}_{}.csv'.format(baseFolder, programName, classifier)
					fileExists = util.pathExists(programFileName)
					if fileExists:
						# Obtém as métricas do programa / classificador
						accuracy, precision, recall, f1 = getMLMetricsFromClassificationFile(programFileName, targetColumn, programName)
						newMutantsMetrics = pd.DataFrame(data=[[programName, accuracy * 100, precision * 100, recall * 100, f1 * 100, classifier]], columns=['ProgramName', 'Accuracy', 'Precision', 'Recall', 'F1', 'Classifier'])
						program_ClassifierMetrics = program_ClassifierMetrics.append(newMutantsMetrics)

				# Verifica qual o melhor parâmetro para aquele programa
				if fileExists:
					bestClassifier = program_ClassifierMetrics.sort_values('F1', ascending=False).head(n=1)['Classifier'].values[0]
					bestFile = '{}/{}_{}.csv'.format(baseFolder, programName, bestClassifier)
					newFile = '{}/{}.csv'.format(baseFolder, programName)

					bestParameter = bestProgram_Classifier_Parameters.query('Column == \'{}\' and Program == \'{}\' and Classifier == \'{}\''.format(targetColumn, programName, bestClassifier))
					if not bestParameter.empty:
						bestParameter = bestParameter['Parameter'].values[0]
					else:
						bestParameter = ''
					newBestProgram_Classifier = pd.DataFrame(data=[[targetColumn, programName, bestClassifier, bestParameter]], columns=['Column', 'Program', 'Classifier', 'Parameter'])
					bestProgram_Classifier = bestProgram_Classifier.append(newBestProgram_Classifier)

					copyfile(bestFile, newFile)
			
			# Escreve o arquivo
			bestProgram_Classifier['Program.UPPER'] = bestProgram_Classifier["Program"].str.upper()
			bestProgram_Classifier = bestProgram_Classifier.sort_values(by=['Column', 'Program.UPPER'])
			del bestProgram_Classifier['Program.UPPER']

			util.writeDataFrameInCsvFile(fileName, bestProgram_Classifier.query('Column == \'{}\''.format(targetColumn)))
		else:
			newBestProgram_Classifier = util.createDataFrameFromCSV(fileName, hasHeader=True, columnIndex=0)
			bestProgram_Classifier = pd.concat([bestProgram_Classifier, newBestProgram_Classifier])

			bestProgram_Classifier['Program.UPPER'] = bestProgram_Classifier["Program"].str.upper()
			bestProgram_Classifier = bestProgram_Classifier.sort_values(by=['Column', 'Program.UPPER'])
			del bestProgram_Classifier['Program.UPPER']

	return bestProgram_Classifier


def getBestParameterForEachClassificationOfPrograms(possibleTargetColumn, possiblePrograms, possibleClassifiers, writeFile = False):

	bestProgram_Classifier_Parameters = pd.DataFrame()

	# Percorre todos as colunas
	for targetColumn in possibleTargetColumn:
		baseFolder = '{}/ML/Results/{}/Classification'.format(os.getcwd(), targetColumn)
		fileName = '{}/bestParameterForEachClassificationOfPrograms.csv'.format(baseFolder)

		if writeFile:

			# Percorre todos os programas
			for programName in possiblePrograms:
				# Percorre todos os classificadores daquele programa
				for classifier in possibleClassifiers:
					program_Classifier_Parameters_Metrics = pd.DataFrame()
					#print('Column: {} | Program: {} | Classifier: {}'.format(targetColumn, programName, classifier))
					
					# Percorre todos os parâmetros daquele programa / classificador
					for parameter in util.getPossibleParameters(classifier):
						programFileName = '{}/{}_{}_{}.csv'.format(baseFolder, programName, classifier, parameter)
						fileExists = util.pathExists(programFileName)
						if fileExists:
							# Obtém as métricas do programa / classificador / parâmetro selecionado
							accuracy, precision, recall, f1 = getMLMetricsFromClassificationFile(programFileName, targetColumn, programName)
							newMutantsMetrics = pd.DataFrame(data=[[programName, accuracy * 100, precision * 100, recall * 100, f1 * 100, parameter]], columns=['ProgramName', 'Accuracy', 'Precision', 'Recall', 'F1', 'SampleSplit'])
							program_Classifier_Parameters_Metrics = program_Classifier_Parameters_Metrics.append(newMutantsMetrics)
							
					
					# Verifica qual o melhor parâmetro para aquele programa / classificador
					if fileExists:
						bestParameter = program_Classifier_Parameters_Metrics.sort_values('F1', ascending=False).head(n=1)['SampleSplit'].values[0]
						bestFile = '{}/{}_{}_{}.csv'.format(baseFolder, programName, classifier, bestParameter)
						newFile = '{}/{}_{}.csv'.format(baseFolder, programName, classifier)

						newBestProgram_Classifier_Parameters = pd.DataFrame(data=[[targetColumn, programName, classifier, bestParameter]], columns=['Column', 'Program', 'Classifier', 'Parameter'])
						bestProgram_Classifier_Parameters = bestProgram_Classifier_Parameters.append(newBestProgram_Classifier_Parameters)

						copyfile(bestFile, newFile)
			
			# Escreve o arquivo
			bestProgram_Classifier_Parameters['Program.UPPER'] = bestProgram_Classifier_Parameters["Program"].str.upper()
			bestProgram_Classifier_Parameters = bestProgram_Classifier_Parameters.sort_values(by=['Column', 'Program.UPPER'])
			del bestProgram_Classifier_Parameters['Program.UPPER']

			util.writeDataFrameInCsvFile(fileName, bestProgram_Classifier_Parameters.query('Column == \'{}\''.format(targetColumn)))
		else:
			newBestProgram_Classifier_Parameters = util.createDataFrameFromCSV(fileName, hasHeader=True, columnIndex=0)
			bestProgram_Classifier_Parameters = pd.concat([bestProgram_Classifier_Parameters, newBestProgram_Classifier_Parameters])
			
			bestProgram_Classifier_Parameters['Program.UPPER'] = bestProgram_Classifier_Parameters["Program"].str.upper()
			bestProgram_Classifier_Parameters = bestProgram_Classifier_Parameters.sort_values(by=['Column', 'Program.UPPER'])
			del bestProgram_Classifier_Parameters['Program.UPPER']

	return bestProgram_Classifier_Parameters