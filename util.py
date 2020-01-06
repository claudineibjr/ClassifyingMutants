#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
from datetime import datetime
import csv
from shutil import copyfile

############
# --- Pandas
import pandas as pd

################
# --- Statistics
from statistics import mean
from statistics import median

def getPossibleClassifiers():
	possibleClassifiers = ['KNN', 'DT', 'RF', 'SVM', 'LDA', 'LR', 'GNB']
	return possibleClassifiers

def getFullNamePossibleClassifiers():
	fullNameClassifiers = {
		'KNN': 'K Nearest Neighbors',
		'DT': 'Decision Tree',
		'RF': 'Random Forest',
		'SVM': 'Support Vector Machine',
		'LDA': 'Linear Discriminant Analysis',
		'LR': 'Logistic Regression',
		'GNB': 'Gaussian Naive Bayes'
	}

	return fullNameClassifiers

def getPossibleTargetColumns():
	possibleTargetColumns = ['MINIMAL', 'EQUIVALENT']
	return possibleTargetColumns

def pathExists(fileName):
    return os.path.exists(fileName)

def createFolder(folderName):
    os.mkdir(folderName)

def getContentFromFile(fileName):
    file = open(fileName, "r")
    content = file.read()
    file.close()

    return content

def formatNow():
    now = datetime.now()
    return now.strftime('%d/%m/%Y %H:%M:%S')      

def writeDataFrameInCsvFile(fileName, dataFrame, sep = ',', mode = 'w'):
    pd.DataFrame(dataFrame).to_csv(fileName, sep=sep, mode=mode)

def writeInCsvFile(fileName, content, header = None, delimiter = ';', mode='w'):
    if header == None:
        with open(fileName, mode=mode) as resultFile:
            resultWriter = csv.writer(resultFile, delimiter = delimiter, quotechar='"', quoting=csv.QUOTE_MINIMAL)
    
            resultWriter.writerows(content)
    else:
        with open(fileName, mode=mode) as resultFile:
            resultWriter = csv.writer(resultFile, delimiter = delimiter, quotechar='"', quoting=csv.QUOTE_MINIMAL)
    
            resultWriter.writerow(header)
            resultWriter.writerows(content)

def write(fileName, content, mode='w'):
    file = open(fileName, mode)
    file.write(content)
    file.close()

def convertStringToArray(content, separator):
    return str(content).split(separator)

def extractResults(baseFolder, destinarionFolder, fileToCopy, concatenateProgramName = False):
    for (subFolder, dirNames, files) in os.walk(baseFolder):
        sourceProgram = "{}.c".format(subFolder[str(subFolder).rfind("/") + 1:])
        
        # Verifica se dentro da pasta tem um arquivo com a extensão "c" com o mesmo nome da pasta
        if list(files).__contains__(sourceProgram):
            programName = sourceProgram.replace(".c", "")
            logFolder = "{baseFolder}/{programName}/log".format(
                baseFolder = baseFolder, programName = programName)

            patternFile = fileToCopy[str(fileToCopy).rfind("/") + 1: ]
            for (_subFolder, _dirNames, _files) in os.walk(logFolder):
                for file in _files:
                    if str(file).__contains__(patternFile):
                        fileNameToCopy = "{baseFolder}/{programName}/log/{file}".format(
                            baseFolder = baseFolder, programName = programName, file = file)
                        if concatenateProgramName:
                            destinationFile = "{destinarionFolder}/{programName}_{file}".format(
                                destinarionFolder = destinarionFolder, 
                                programName = programName.replace(".c", ""), file = file)
                        else:
                            destinationFile = "{destinarionFolder}/{file}".format(
                                destinarionFolder = destinarionFolder, file = file)

                        #print(fileNameToCopy, destinationFile)
                        copyfile(fileNameToCopy, destinationFile)

def deleteResults(baseFolder, fileToDelete):
    for (subFolder, dirNames, files) in os.walk(baseFolder):
        sourceProgram = "{}.c".format(subFolder[str(subFolder).rfind("/") + 1:])
        
        # Verifica se dentro da pasta tem um arquivo com a extensão "c" com o mesmo nome da pasta
        if list(files).__contains__(sourceProgram):
            programName = sourceProgram.replace(".c", "")
            logFolder = "{baseFolder}/{programName}/log".format(
                baseFolder = baseFolder, programName = programName)

            patternFile = fileToDelete[str(fileToDelete).rfind("/") + 1: ]
            for (_subFolder, _dirNames, _files) in os.walk(logFolder):
                for file in _files:
                    if str(file).__contains__(patternFile):
                        fileNameToDelete = "{baseFolder}/{programName}/log/{file}".format(
                            baseFolder = baseFolder, programName = programName, file = file)
                        
                        os.remove(fileNameToDelete)

def getPrograms(folder = '{}/Programs'.format(os.getcwd())):
    folders = []
    
    # Percorre todas as pastas dentro do diretório base
    for (subFolder, dirNames, files) in os.walk(folder):

        # Pega o nome da pasta e concatenca com ".c" para que identifique o 
        #   (A sessão de teste, o programa (".c") e a pasta devem ter o mesmo nome)
        sourceProgram = "{}.c".format(subFolder[str(subFolder).rfind("/") + 1:])
        
        # Verifica se dentro da pasta tem um arquivo com a extensão "c" com o mesmo nome da pasta
        if list(files).__contains__(sourceProgram):
            folders.append(subFolder)

    return folders

def splitFileInColumns(fileName, separator = ','):
    contentFile = getContentFromFile(fileName)
    return [line.split(separator) for line in contentFile.splitlines()]

def getFoldersInFolder(folder):
    folders = []
    
    # Percorre todas as pastas dentro do diretório base
    for (subFolder, dirNames, files) in os.walk(folder):
        if str(folder).count('/') +1 == str(subFolder).count('/'):
            folders.append(subFolder)

    return folders

def createDataFrameFromCSV(csvFile, hasHeader = False, separator = ','):
    if hasHeader:
        return pd.read_csv(csvFile, index_col=0, sep=separator)
    else:
        return pd.read_csv(csvFile, sep=separator)

def getFilesInFolder(folder):
    files = []

    for file in os.listdir(folder):
        fullFilePath = '{}/{}'.format(folder, file)
        if os.path.isfile(fullFilePath):
            files.append(fullFilePath)

    return files

def getPathName(fullPath):
    return str(fullPath)[str(fullPath).rindex('/') + 1 : ]

def getPreviousFolder(fullPath):
    return str(fullPath)[ 0: str(fullPath).rindex('/')]

def normalize(data):
    maxValue = max(data)
    minValue = min(data)

    return [ (_data - minValue) / (maxValue - minValue) for _data in data]

def renameFolder(oldName, newName):
    os.rename(oldName, newName)

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
    print(getPathName('/home/claudinei/Repositories/RelationshipBetweenMutationAndGFC/ML/Results/MINIMAL/Classification/Heap.csv'))