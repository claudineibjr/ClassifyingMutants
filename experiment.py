#!/usr/bin/python
# -*- coding: utf-8 -*-

####################################################################################################
#										   prot2poke                                               #
####################################################################################################
# Descricao: Este script transforma um arquivo GFC gerado pela ProteumIM em um arquivo GFC no      #
#	 		 do gerador de arcos primitivos                                                        #
#																			   					   #
# Autor: Claudinei Brito Junior (claudineibjr | @hotmail.com, @gmail.com, @usp.br)                 #
#																			   					   #
# prot2poke <file-name>							      			   			                       #
# - prog-name: nome do arquivo GFC 							   				   	                   #
#	Exemplo: prot2poke cal.gfc											   			               #
#																				      			   #
# Ultima modificação: 23/11/2018 												       			   #
####################################################################################################

import minimal
import os
import util
import sys

if __name__ == '__main__':
    if len(sys.argv) > 1:
        executionMode = sys.argv[1]
    else:
        print ('##### Exit #####')
        exit()
    
    #executionMode |    1 - Run and analyze
    #                   2 - Run
    #                   3 - Analyze

    # Seta o diretório base de onde deverão estar os programas
    baseExperimentFolder = "{}/Programs_".format(os.getcwd())

    # Percorre todas as pastas dentro do diretório base
    for (subFolder, dirNames, files) in os.walk(baseExperimentFolder):

        # Pega o nome da pasta e concatenca com ".c" para que identifique o 
        #   (A sessão de teste, o programa (".c") e a pasta devem ter o mesmo nome)
        sourceProgram = "{}.c".format(subFolder[str(subFolder).rfind("/") + 1:])
        
        # Verifica se dentro da pasta tem um arquivo com a extensão "c" com o mesmo nome da pasta
        if list(files).__contains__(sourceProgram):
            print ('### BEGIN ###')
            print ('##########\t   Executing ' + sourceProgram + '\t ' + util.formatNow() + '\t   ##########')

            # Faz a execução do experimento passando como parâmetro a pasta desejada
            minimal.main(baseExperimentFolder, subFolder, executionMode)