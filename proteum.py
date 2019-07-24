#!/usr/bin/python
# -*- coding: utf-8 -*-

import subprocess

'''
    Esta função irá criar a sessão de teste com base nos parâmetros enviados
'''
def createSession(baseFolder, sessionName, executionType, sourceFile, executableFile, directory, driver):

    compilationCommand = "gcc {sourceFile}.c {driver} -o {executableFile} -lm -w".format(
        sourceFile = sourceFile, executableFile = executableFile, driver = driver)
    command = "cd {baseFolder}; {compilationCommand}".format(baseFolder = baseFolder, compilationCommand = compilationCommand)
    
    subprocess.call(command, shell=True)

    command = "test-new"

    if (len(executionType) > 0):
        command = "{} -{}".format(command, executionType)

    if (len(sourceFile) > 0):
        command = "{} -S {}".format(command, sourceFile)

    if (len(executableFile) > 0):
        command = "{} -E {}".format(command, executableFile)

    if (len(directory) > 0):
        command = "{} -D {}".format(command, directory)        

    if (len(compilationCommand) > 0):
        command = "{} -C \"{}\"".format(command, compilationCommand)

    if (len(sessionName) > 0):
        command = "{} {}".format(command, sessionName)
    
    #print(command)
    subprocess.run(command, shell=True)

def createTestCases(sessionName, directory, values):
    for value in values:
        createTestCase(sessionName, directory, value)

'''
    Esta função irá adicionar casos de teste à sessão de teste
'''
def createTestCase(sessionName, directory, value):
    #tcase -add {value} -D {directory}
    
    command = "tcase -add -p \"{value}\" -trace -D {directory} {sessionName}".format(
        value = value, directory = directory, sessionName = sessionName)

    #print(command)
    subprocess.call(command, shell=True)

'''
    Esta função irá exibir os casos de teste da sessão de teste
'''
def showTestCases(sessionName, directory):
    command = "tcase -l -D {directory} {sessionName}".format(directory = directory, sessionName = sessionName)

    #print(command)
    subprocess.call(command, shell=True)

'''
    Esta função irá deletar todos os casos de teste existentes na sessão de teste
'''
def deleteTestCase(sessionName, directory):
    command = "tcase -d -D {directory} {sessionName}".format(directory = directory, sessionName = sessionName)

    #print(command)
    subprocess.call(command, shell=True)

'''
    Esta função irá criar um set limpo de mutantes (excluindo os existentes)
'''
def createEmptySetMutants(sessionName, directory):
    command = "muta -create -D {directory} {sessionName}".format(
        directory = directory, sessionName = sessionName)

    #print(command)
    subprocess.call(command, shell=True)

'''
    Esta função irá gerar mutates para 100% todos os operadores, permitindo gerar um número ilimitado de mutantes por ponto de mutação
'''
def generateMutants(sessionName, directory, unitName):
    command = "muta-gen -unit {unitName} -u- 1.0 0 -D {directory} {sessionName}".format(
        directory = directory, sessionName = sessionName, unitName = unitName)

    #print(command)
    subprocess.call(command, shell=True)

'''
    Esta função irá executar os mutantes "contra" os casos de teste
'''
def executeMutants(sessionName, directory):
    command = "exemuta -exec -v . -D {directory} -trace {sessionName}".format(
        directory = directory, sessionName = sessionName)

    #print(command)
    subprocess.call(command, shell=True)

'''
    Esta função irá exibir os mutantes na tela
'''
def showMutants(sessionName, directory, mutantNumber):
    if (len(mutantNumber) > 0):
        mutants = "\"{mutantNumber}\"".format(mutantNumber = mutantNumber)
    else:
        mutants = ""

    outputFile = "{directory}/log/mutants.txt".format(directory = directory)

    command = "muta -l {selectMutant} {mutants} -D {directory} {sessionName} > {outputFile}".format(
        directory = directory, sessionName = sessionName, 
        mutants = mutants, selectMutant = "-x" if len(mutants) > 0 else "",
        outputFile = outputFile)

    #print(command)
    subprocess.call(command, shell=True)      

'''
    Esta função irá desabilitar, excluir ou apenas listar os casos de teste não efetivos
'''
def listGood(sessionName, directory, option):
    #option = "i" Disable non-effective and list effective
    #option = "d" Delete non-effective and list effective
    #option = "" List effective

    command = "list-good {hiphen}{option} -research -D {directory} {sessionName}".format(
        directory = directory, sessionName = sessionName, hiphen = "-" if len(option) > 0 else "", option = option)

    #print(command)
    subprocess.call(command, shell=True)

'''
    Esta função irá setar um determinado mutante como equivalente (caso esteja vivo)
'''
def setEquivalent(sessionName, directory, mutantNumber):
    if (len(mutantNumber) > 0):
        mutants = "\"{mutantNumber}\"".format(mutantNumber = mutantNumber)
    else:
        mutants = ""

    command = "muta -equiv {selectMutant} {mutants} -D {directory} {sessionName}".format(
        directory = directory, sessionName = sessionName, 
        mutants = mutants, selectMutant = "-x" if len(mutants) > 0 else "")

    #print(command)
    subprocess.call(command, shell=True)    

'''
    Esta função irá criar relatórios acerca da sessão de testes
'''
def generateReport(sessionName, directory):
    command = "report -tcase -L 511 -D {directory} {sessionName}".format(
        directory = directory, sessionName = sessionName)

    #print(command)
    subprocess.call(command, shell=True)

    command = "report -trace -L 2 -D {directory} {sessionName}".format(
        directory = directory, sessionName = sessionName)

    subprocess.call(command, shell=True)

    #print(command)
    subprocess.call(command, shell=True)   