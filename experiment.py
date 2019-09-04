#!/usr/bin/python
# -*- coding: utf-8 -*-

import minimal
import os
import util
import sys

if __name__ == '__main__':
    if len(sys.argv) > 1:
        executionMode = sys.argv[1]
    else:
        print ('Please specify the execution mode. 1 For run and analyze | 2 For run | 3 For analyze')
        print ('##### Exit #####')
        exit()
    
    #executionMode |    1 - Run and analyze
    #                   2 - Run
    #                   3 - Analyze

    # Seta o diretório base de onde deverão estar os programas
    baseExperimentFolder = "{}/Programs".format(os.getcwd())

    # Percorre todas as pastas dentro do diretório base
    programsFolder = util.getPrograms(baseExperimentFolder)
    for subFolder in programsFolder:
        sourceProgram = '{}.c'.format(util.getFolderName(subFolder))
        print ('### BEGIN ###')
        print ('##########\t   Executing ' + sourceProgram + '\t ' + util.formatNow() + '\t   ##########')

        # Faz a execução do experimento passando como parâmetro a pasta desejada
        minimal.main(baseExperimentFolder, subFolder, executionMode)