# Statistics
from statistics import mean
from statistics import median

import util
import os

def getProgramsInfo():
	programsInfoFileName = '{}/Programs/ProgramsInfo.csv'.format(os.getcwd())
	programsInfo = util.getContentFromFile(programsInfoFileName)

	return [line.split(',') for line in programsInfo.splitlines()]

def getProgramInfo(programsInfo, programName):
	for program in programsInfo:
		if program[0] == programName:
			return program

	return None

def analyzeProgramAProgram(targetColumn, possibleTargetColumns, possibleClassifiers, programsInfo):
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
				
				minAccuracy.append(		float(columns[0][1]))
				minPrecision.append(	float(columns[0][2]))
				minRecall.append(		float(columns[0][3]))
				minF1.append(			float(columns[0][4]))
				
				maxAccuracy.append(		float(columns[1][1]))
				maxPrecision.append(	float(columns[1][2]))
				maxRecall.append(		float(columns[1][3]))
				maxF1.append(			float(columns[1][4]))
				
				meanAccuracy.append(	float(columns[2][1]))
				meanPrecision.append(	float(columns[2][2]))
				meanRecall.append(		float(columns[2][3]))
				meanF1.append(			float(columns[2][4]))
				
				medianAccuracy.append(	float(columns[3][1]))
				medianPrecision.append(	float(columns[3][2]))
				medianRecall.append(	float(columns[3][3]))
				medianF1.append(		float(columns[3][4]))

				programData.append([program, float(columns[1][1]), float(columns[1][2]), float(columns[1][3]), float(columns[1][4])])

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

if __name__ == '__main__':
	programsInfo = getProgramsInfo()
	
	possibleTargetColumns = ['MINIMAL', 'EQUIVALENT']
	possibleClassifiers = ['KNN', 'DT', 'RF', 'SVM']

	analyzeProgramAProgram('MINIMAL', possibleTargetColumns, possibleClassifiers, programsInfo)
	analyzeProgramAProgram('EQUIVALENT', possibleTargetColumns, possibleClassifiers, programsInfo)