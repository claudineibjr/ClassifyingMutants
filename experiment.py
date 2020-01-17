#!/usr/bin/python
# -*- coding: utf-8 -*-

import genMutants
import os
import util
import sys

if __name__ == '__main__':   
    # --mode    |   1 - Run and analyze
    #               2 - Run
    #               3 - Analyze

    # --allPrograms
    # --program

    executionMode = None
    allPrograms = False
    program = None

    # Trought into all parameters
    arguments = sys.argv
    for iCount in range(1, len(arguments), 1):
        arg = arguments[iCount]
        if arg == '--mode':
            executionMode = int(arguments[iCount + 1])
        elif arg == '--allPrograms':
            allPrograms = True
        elif arg == '--program':
            program = arguments[iCount + 1]

    if executionMode is None:
        print ('Please specify the execution mode trought --mode parameter. 1 For run and analyze | 2 For run | 3 For analyze')
        print ('##### Exit #####')
        sys.exit()

    if allPrograms == False and program is None:
        print ('Please specify the program to be executed through --program parameter or execute all through --allPrograms parameter')
        print ('##### Exit #####')
        sys.exit()

    # Seta o diretório base de onde deverão estar os programas
    baseExperimentFolder = "{}/Programs".format(os.getcwd())

    # Percorre todas as pastas dentro do diretório base
    programsFolder = util.getPrograms(baseExperimentFolder) if allPrograms else ['{}/{}'.format(baseExperimentFolder, program)]
    for subFolder in programsFolder:

        sourceProgram = '{}.c'.format(util.getPathName(subFolder))
        print ('### BEGIN ###')
        print ('##########\t   Executing ' + sourceProgram + '\t ' + util.formatNow() + '\t   ##########')

        # Faz a execução do experimento passando como parâmetro a pasta desejada
        genMutants.main(baseExperimentFolder, subFolder, executionMode)