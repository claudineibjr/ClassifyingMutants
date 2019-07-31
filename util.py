#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
from datetime import datetime
import csv
from shutil import copyfile

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
                        

def normalize(data):
    maxValue = max(data)
    minValue = min(data)

    return [ (_data - minValue) / (maxValue - minValue) for _data in data]