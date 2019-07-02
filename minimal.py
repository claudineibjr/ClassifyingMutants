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

import sys

from Pedro.ComputeMinimalMuta import computeMinimal
from Pedro.prot2poke import prot2PokeMain

import proteum
import util
import gfcUtils

_IM_NUMBER = 0  # Número do mutante
_IM_RESULT = 1  # Resultado (morto, vivo, equivalente)
_IM_STATUS = 2  # Status do mutante (ativo, inativo)
_IM_OPERATOR = 3 # Operador
_IM_PROGRAM_GRAPH_NODE = 4 # Nó da mutação
_IM_MINIMAL = 5 # É minimal?
_IM_PRIMITIVE_ARC = 6 # É origem ou destino de arco primitivo?
_IM_SOURCE_PRIMITIVE_ARC = 7 # É origem de arco primitivo?
_IM_TARGET_PRIMITIVE_ARC = 8 # É destino de arco primitivo?
_IM_EQUIVALENT = 9 # É equivalente?
_IM_DISTANCE_BEGIN = 10 # Distâncias do nó inicial
_IM_DISTANCE_BEGIN_MIN = 11 # Distância mínima do nó inicial
_IM_DISTANCE_BEGIN_MAX = 12 # Distância máxima do nó inicial
_IM_DISTANCE_BEGIN_AVG = 13 # Distância média do nó inicial
_IM_DISTANCE_END = 14 # Distâncias do nó final
_IM_DISTANCE_END_MIN = 15 # Distância mínima do nó final
_IM_DISTANCE_END_MAX = 16 # Distância máxima do nó final
_IM_DISTANCE_END_AVG = 17 # Distância média do nó final
_IM_SOURCE_NODE = 18 # Nós origem
_IM_TARGET_NODE = 19 # Nós destino
_IM_SOURCE_NODE_PRIMITIVE = 20 # Algum dos nós de origem é origem ou destino do arco primitivo?
_IM_TARGET_NODE_PRIMITIVE = 21 # Algum dos nós de destino é origem ou destino do arco primitivo?

_IMA_MUTANTS = 0    #Representa o número total de mutantes
_IMA_MINIMALS = 1   # Representa o número total de mutantes minimais
_IMA_EQUIVALENTS = 2    # Representa o número total de mutantes equivalentes
_IMA_PRIMITIVE_ARC = 3  # Representa o número total de arcos primitivos
_IMA_SOURCE_PRIMITIVE_ARC = 4   # Representa as origens dos arcos primitivos
_IMA_TARGET_PRIMITIVE_ARC = 5   # Representa os destinos dos arcos primitivos
_IMA_MINIMALS_PRIMITIVE_ARC = 6 # Representa os minimais nos arcos primitivos
_IMA_MINIMALS_SOURCE_PRIMITIVE_ARC = 7  # Representa os minimais na origem dos arcos primitivos
_IMA_MINIMALS_TARGET_PRIMITIVE_ARC = 8  # Representa os minimais no destino dos arcos primitivos
_IMA_MINIMALS_NODE_SOURCE_OR_TARGET_PRIMITIVE = 9 # Representa os minimais nos arcos primitivos ou minimais com origens ou destinos pertencentes aos arcos primitivos
_IMA_NON_MINIMALS_NODE_SOURCE_OR_TARGET_PRIMITIVE = 10 # Representa os minimais nos arcos primitivos ou minimais com origens ou destinos pertencentes aos arcos primitivos
_IMA_EQUIVALENTS_PRIMITIVE_ARC = 11 # Representa os equivalentes nos arcos primitivos
_IMA_EQUIVALENTS_SOURCE_PRIMITIVE_ARC = 12  # Representa os equivalentes na origem dos arcos primitivos
_IMA_EQUIVALENTS_TARGET_PRIMITIVE_ARC = 13  # Representa os equivalentes no destino dos arcos primitivos
_IMA_EQUIVALENTS_NODE_SOURCE_OR_TARGET_PRIMITIVE = 14   # Representa os equivalentes nos arcos primitivos ou equivalentes com origens ou destinos pertencentes aos arcos primitivos
_IMA_NON_EQUIVALENTS_NODE_SOURCE_OR_TARGET_PRIMITIVE = 15   # Representa os equivalentes nos arcos primitivos ou equivalentes com origens ou destinos pertencentes aos arcos primitivos
_IMA_MINIMALS_SOURCES_ARE_PRIMITIVE_ARC = 16    # Representa os minimais com nós origens pertencentes aos arcos primitivos
_IMA_MINIMALS_TARGETS_ARE_PRIMITIVE_ARC = 17    # Representa os minimais com nós destinos pertencentes aos arcos primitivos
_IMA_EQUIVALENTS_SOURCES_ARE_PRIMITIVE_ARC = 18 # Representa os equivalentes com nós origens pertencentes aos arcos primitivos
_IMA_EQUIVALENTS_TARGETS_ARE_PRIMITIVE_ARC = 19 # Representa os equivalentes com nós destinos pertencentes aos arcos primitivos

def getMinimalMutants(baseFolder, sourceFile):
    fileName = "{}/log/minimal.txt".format(baseFolder)
    #fileName = "{}/arc_prim/minimal.txt".format(baseFolder) #Utilizar as informações dos minimais que o Pedro já passou

    if (not util.pathExists(fileName)):
        computeMinimal("{baseFolder}/{sourceFile}".format(
            baseFolder = baseFolder, sourceFile = sourceFile), False,
            "{baseFolder}/log".format(
                baseFolder = baseFolder))

    contentFile = util.getContentFromFile(fileName)

    minimalMutants = []

    linesFile = contentFile.splitlines()
    for line in linesFile:
        line = line.replace("\n", "")
        if len(line) > 0:
            minimalMutants.append(line)

    return minimalMutants     

'''
    Função principal que executa todo o passo a passo na Proteum para geração dos mutantes
'''
def executeProteum(baseFolder, sourceFile, sessionName, executableFile, executionType, directory, unitName):
    if not util.pathExists("{}/log".format(baseFolder)):
        util.createFolder("{}/log".format(baseFolder))
    
    #####################
    # Create Test Session
    #####################
    print ('\n##### \tCriando sessão de testes ' + util.formatNow() + '\t#####'    )
    proteum.createSession(baseFolder, sessionName, executionType, sourceFile, executableFile, directory)

    ###################
    # Create Test Cases
    ###################
    print ('\n##### \tCriando casos de teste ' + util.formatNow() + '\t#####'      )
    testCases = util.convertStringToArray(util.getContentFromFile("{}/testset.txt".format(baseFolder)), "\n")
    proteum.createTestCases(sessionName, directory, testCases)

    #################
    # Show Test Cases
    #################
    print ('\n##### \tCasos de teste ' + util.formatNow() + '\t#####'              )
    print(proteum.showTestCases(sessionName, directory))

    ###################
    # Delete Test Cases
    ###################
    #print ('\n##### \tCasos de teste deletados ' + formatNow() + '\t#####')
    #proteum.deleteTestCase(sessionName, directory)

    ##################
    # Generate mutants
    ##################
    print ('\n##### \tGerando mutantes ' + util.formatNow() + '\t#####'            )
    proteum.createEmptySetMutants(sessionName, directory)
    proteum.generateMutants(sessionName, directory, unitName)

    #################
    # Execute mutants
    #################
    print ('\n##### \tExecutando mutantes ' + util.formatNow() + '\t#####'         )
    proteum.executeMutants(sessionName, directory)

    ###############################
    # Seta mutante como equivalente
    ###############################
    print ('\n##### \tSetando mutantes como equivalentes ' + util.formatNow() + '\t#####'         )
    proteum.setEquivalent(sessionName, directory, "")    

    #########################################
    # Execute mutants considering equivalents
    #########################################
    print ('\n##### \tExecutando mutantes considerando os equivalentes ' + util.formatNow() + '\t#####'         )
    proteum.executeMutants(sessionName, directory)

    #########################
    # Casos de teste efetivos
    #########################
    print ('\n##### \tExibindo casos de teste efetivos ' + util.formatNow() + '\t#####'         )
    proteum.listGood(sessionName, directory, "i")

    ####################
    # Gera os relatórios
    ####################
    print ('\n##### \tGeração dos relatórios ' + util.formatNow() + '\t#####'         )
    proteum.showMutants(sessionName, directory, "")

    ###################
    # Exibe os mutantes
    ###################
    print ('\n##### \tExibe mutantes ' + util.formatNow() + '\t#####'         )
    proteum.generateReport(sessionName, directory)

def getMutantsInfo(baseFolder, minimalMutants, sessionName, unitName):
    arrMutantsInfo = []
    arrHeaderMutants = []
    
    mutantsInfoFileName = "{baseFolder}/log/mutants.txt".format(baseFolder = baseFolder)
    #mutantsInfoFileName = "{baseFolder}/arc_prim/mutants.txt".format(baseFolder = baseFolder) #Analisa as informações dos mutantes que o Pedro forneceu
    if (not util.pathExists(mutantsInfoFileName)):
        proteum.showMutants(sessionName, baseFolder, "")

    mutantsInfoFile = util.getContentFromFile(mutantsInfoFileName)

    #gfcFileName = "{baseFolder}/{unitName}.gfc".format(baseFolder = baseFolder, unitName = unitName)
    gfcFileName = "{baseFolder}/arc_prim/{unitName}.gfc".format(baseFolder = baseFolder, unitName = unitName) #Utiliza as informações passadas pelo Pedro
    arcPrimFileName = "{baseFolder}/arc_prim/{unitName}.tes".format(baseFolder = baseFolder, unitName = unitName)
    gfc, numNodes = gfcUtils.gfcMain(gfcFileName, arcPrimFileName)

    mutants = mutantsInfoFile.split("#")
    for mutant in mutants:
        mutantInfos = mutant.splitlines()

        result = ""
        status = ""
        operator = ""
        programGraphNode = ""
        mutantNumber = ""
        minimal = False
        primitiveArc = False
        sourcePrimitiveArc = False
        targetPrimitiveArc = False
        equivalent = False
        distanceBegin = ""
        distanceBegin_min = ""
        distanceBegin_max = ""
        distanceBegin_avg = ""
        distanceEnd = ""
        distanceEnd_min = ""
        distanceEnd_max = ""
        distanceEnd_avg = ""
        sourceNode = ""
        targetNode = ""
        sourcesNodeIsPrimitive = ""
        targetsNodeIsPrimitive = ""

        for mutantInfo in mutantInfos:
            mutantInfo = mutantInfo.strip()

            if len(mutantInfo) > 0:
                if mutantInfo.__contains__("Status"):
                    result = mutantInfo[7:]
                    result = result[0: result.find(" ")]
                    status = mutantInfo[mutantInfo.find(" ", 8) + 1: ]

                    equivalent = result.__contains__("Equivalent")

                elif mutantInfo.__contains__("Operator"):
                    operator = mutantInfo[mutantInfo.find("(") + 1: mutantInfo.find(")")]
                elif mutantInfo.__contains__("Program graph node"):
                    programGraphNode = mutantInfo[mutantInfo.find(":") + 2:]

                    _source, _target, \
                        _distanceFromBegin, _distanceFromBegin_min, _distanceFromBegin_max, _distanceFromBegin_avg,\
                            _distanceFromEnd, _distanceFromEnd_min, _distanceFromEnd_max, _distanceFromEnd_avg,\
                                 _arcPrimSource, _arcPrimTarget, _sourcesNodeIsPrimitive, _targetsNodeIsPrimitive = gfcUtils.getInfoNode(gfc, programGraphNode, numNodes)
                    
                    distanceBegin = _distanceFromBegin
                    distanceBegin_min = _distanceFromBegin_min
                    distanceBegin_max = _distanceFromBegin_max
                    distanceBegin_avg = _distanceFromBegin_avg
                    
                    distanceEnd = _distanceFromEnd
                    distanceEnd_min = _distanceFromEnd_min
                    distanceEnd_max = _distanceFromEnd_max
                    distanceEnd_avg = _distanceFromEnd_avg

                    sourceNode = _source
                    targetNode = _target
                    sourcePrimitiveArc = _arcPrimSource
                    targetPrimitiveArc = _arcPrimTarget
                    primitiveArc = sourcePrimitiveArc or targetPrimitiveArc

                    sourcesNodeIsPrimitive = _sourcesNodeIsPrimitive
                    targetsNodeIsPrimitive = _targetsNodeIsPrimitive

                else:
                    if mutantInfo.isnumeric():
                        mutantNumber = mutantInfo
                        
                        if mutantNumber in minimalMutants:
                            minimal = True
                        else:
                            minimal = False
        
        if len(mutantNumber.strip()) > 0:            
            arrMutantInfo = []
            arrMutantInfo.append(mutantNumber)
            arrMutantInfo.append(result)
            arrMutantInfo.append(status)
            arrMutantInfo.append(operator)
            arrMutantInfo.append(programGraphNode)
            arrMutantInfo.append("1" if minimal else "0")
            arrMutantInfo.append("1" if primitiveArc else "0")
            arrMutantInfo.append("1" if sourcePrimitiveArc else "0")
            arrMutantInfo.append("1" if targetPrimitiveArc else "0")
            arrMutantInfo.append("1" if equivalent else "0")
            arrMutantInfo.append(distanceBegin)
            arrMutantInfo.append(distanceBegin_min)
            arrMutantInfo.append(distanceBegin_max)
            arrMutantInfo.append(distanceBegin_avg)
            arrMutantInfo.append(distanceEnd)
            arrMutantInfo.append(distanceEnd_min)
            arrMutantInfo.append(distanceEnd_max)
            arrMutantInfo.append(distanceEnd_avg)
            arrMutantInfo.append(sourceNode)
            arrMutantInfo.append(targetNode)
            arrMutantInfo.append(sourcesNodeIsPrimitive)
            arrMutantInfo.append(targetsNodeIsPrimitive)

            arrMutantsInfo.append(arrMutantInfo)

    arrHeaderMutants.append("#")
    arrHeaderMutants.append("Result")
    arrHeaderMutants.append("Status")
    arrHeaderMutants.append("Operator")
    arrHeaderMutants.append("Program Graph Node")
    arrHeaderMutants.append("Minimal?")
    arrHeaderMutants.append("Primitive Arc?")
    arrHeaderMutants.append("Source Primitive Arc?")
    arrHeaderMutants.append("Target Primitive Arc?")
    arrHeaderMutants.append("Equivalent?")
    arrHeaderMutants.append("Distance Begin")
    arrHeaderMutants.append("Distance Begin (min)")
    arrHeaderMutants.append("Distance Begin (max)")
    arrHeaderMutants.append("Distance Begin (avg)")
    arrHeaderMutants.append("Distance End")
    arrHeaderMutants.append("Distance End (min)")
    arrHeaderMutants.append("Distance End (max)")
    arrHeaderMutants.append("Distance End (avg)")
    arrHeaderMutants.append("Source Node")
    arrHeaderMutants.append("Target Node")
    arrHeaderMutants.append("Sources Nodes are primitive?")
    arrHeaderMutants.append("Targets Nodes are primitive?")
    
    return arrHeaderMutants, arrMutantsInfo

def computeAdicionalMutantsInfo(mutantsInfo):

    addMutantsInfo = []

    #Basic Info
    totalMinimal = list(filter(lambda info: info[_IM_MINIMAL] == '1', mutantsInfo))
    totalNonMinimal = list(filter(lambda info: info[_IM_MINIMAL] == '0', mutantsInfo))
    totalEquivalents = list(filter(lambda info: info[_IM_EQUIVALENT] == '1', mutantsInfo))
    totalNonEquivalents = list(filter(lambda info: info[_IM_EQUIVALENT] == '0', mutantsInfo))
    totalPrimitiveArc = list(filter(lambda info: info[_IM_PRIMITIVE_ARC] == '1', mutantsInfo))
    totalSourcePrimitiveArc = list(filter(lambda info: info[_IM_SOURCE_PRIMITIVE_ARC] == '1', mutantsInfo))
    totalTargetPrimitiveArc = list(filter(lambda info: info[_IM_TARGET_PRIMITIVE_ARC] == '1', mutantsInfo))

    numTotalMutants = len(mutantsInfo)
    numTotalMinimal = len(totalMinimal)
    numTotalEquivalents = len(totalEquivalents)
    numTotalPrimitiveArc = len(totalPrimitiveArc)
    numTotalSourcePrimitiveArc = len(totalSourcePrimitiveArc)
    numTotalTargetPrimitiveArc = len(totalTargetPrimitiveArc)

    addMutantsInfo.append(numTotalMutants)              #_IMA_MUTANTS
    addMutantsInfo.append(numTotalMinimal)              #_IMA_MINIMALS
    addMutantsInfo.append(numTotalEquivalents)          #_IMA_EQUIVALENTS
    addMutantsInfo.append(numTotalPrimitiveArc)         #_IMA_PRIMITIVE_ARC
    addMutantsInfo.append(numTotalSourcePrimitiveArc)   #_IMA_SOURCE_PRIMITIVE_ARC
    addMutantsInfo.append(numTotalTargetPrimitiveArc)   #_IMA_TARGET_PRIMITIVE_ARC

    #Additional info
    minimalInPrimitiveArc = list(filter(lambda info: info[_IM_PRIMITIVE_ARC] == '1', totalMinimal))
    minimalInSourcePrimitiveArc = list(filter(lambda info: info[_IM_SOURCE_PRIMITIVE_ARC] == '1', totalMinimal))
    minimalInTargetPrimitiveArc = list(filter(lambda info: info[_IM_TARGET_PRIMITIVE_ARC] == '1', totalMinimal))
    equivalentInPrimitiveArc = list(filter(lambda info: info[_IM_PRIMITIVE_ARC] == '1', totalEquivalents))
    equivalentInSourcePrimitiveArc = list(filter(lambda info: info[_IM_SOURCE_PRIMITIVE_ARC] == '1', totalEquivalents))
    equivalentInTargetPrimitiveArc = list(filter(lambda info: info[_IM_TARGET_PRIMITIVE_ARC] == '1', totalEquivalents))
    
    minimalInPrimitiveArcOrWithSourceOrTargetInPrimitive = list(filter(lambda info:
        info[_IM_PRIMITIVE_ARC] == '1' or
        list(info[_IM_SOURCE_NODE_PRIMITIVE]).__contains__(1) or 
        list(info[_IM_TARGET_NODE_PRIMITIVE]).__contains__(1),
    totalMinimal))
    nonMinimalInPrimitiveArcOrWithSourceOrTargetInPrimitive = list(filter(lambda info:
        info[_IM_PRIMITIVE_ARC] == '1' or
        list(info[_IM_SOURCE_NODE_PRIMITIVE]).__contains__(1) or 
        list(info[_IM_TARGET_NODE_PRIMITIVE]).__contains__(1),
    totalNonMinimal))
    minimalWithSourceNodeInPrimitiveArc = list(filter(lambda info: list(info[_IM_SOURCE_NODE_PRIMITIVE]).__contains__(1), totalMinimal))
    minimalWithTargetNodeInPrimitiveArc = list(filter(lambda info: list(info[_IM_TARGET_NODE_PRIMITIVE]).__contains__(1), totalMinimal))
    
    equivalentInPrimitiveArcOrWithSourceOrTargetInPrimitive = list(filter(lambda info:
        info[_IM_PRIMITIVE_ARC] == '1' or
        list(info[_IM_SOURCE_NODE_PRIMITIVE]).__contains__(1) or 
        list(info[_IM_TARGET_NODE_PRIMITIVE]).__contains__(1),
    totalEquivalents))
    nonEquivalentInPrimitiveArcOrWithSourceOrTargetInPrimitive = list(filter(lambda info:
        info[_IM_PRIMITIVE_ARC] == '1' or
        list(info[_IM_SOURCE_NODE_PRIMITIVE]).__contains__(1) or 
        list(info[_IM_TARGET_NODE_PRIMITIVE]).__contains__(1),
    totalNonEquivalents))    
    equivalentWithSourceNodeInPrimitiveArc = list(filter(lambda info: list(info[_IM_SOURCE_NODE_PRIMITIVE]).__contains__(1), totalEquivalents))
    equivalentWithTargetNodeInPrimitiveArc = list(filter(lambda info: list(info[_IM_TARGET_NODE_PRIMITIVE]).__contains__(1), totalEquivalents))

    numMinimalInPrimitiveArc = len(minimalInPrimitiveArc)
    numMinimalInSourcePrimitiveArc = len(minimalInSourcePrimitiveArc)
    numMinimalInTargetPrimitiveArc = len(minimalInTargetPrimitiveArc)
    numEquivalentInPrimitiveArc = len(equivalentInPrimitiveArc)
    numEquivalentInSourcePrimitiveArc = len(equivalentInSourcePrimitiveArc)
    numEquivalentInTargetPrimitiveArc = len(equivalentInTargetPrimitiveArc)

    numMinimalInPrimitiveArcOrWithSourceOrTargetInPrimitive = len(minimalInPrimitiveArcOrWithSourceOrTargetInPrimitive)
    numNonMinimalInPrimitiveArcOrWithSourceOrTargetInPrimitive = len(nonMinimalInPrimitiveArcOrWithSourceOrTargetInPrimitive)
    numMinimalWithSourceNodeInPrimitiveArc = len(minimalWithSourceNodeInPrimitiveArc)
    numMinimalWithTargetNodeInPrimitiveArc = len(minimalWithTargetNodeInPrimitiveArc)

    numEquivalentInPrimitiveArcOrWithSourceOrTargetInPrimitive = len(equivalentInPrimitiveArcOrWithSourceOrTargetInPrimitive)
    numNonEquivalentInPrimitiveArcOrWithSourceOrTargetInPrimitive = len(nonEquivalentInPrimitiveArcOrWithSourceOrTargetInPrimitive)
    numEquivalentWithSourceNodeInPrimitiveArc = len(equivalentWithSourceNodeInPrimitiveArc)
    numEquivalentWithTargetNodeInPrimitiveArc = len(equivalentWithTargetNodeInPrimitiveArc)
                        

    addMutantsInfo.append(numMinimalInPrimitiveArc)                     #_IMA_MINIMALS_PRIMITIVE_ARC
    addMutantsInfo.append(numMinimalInSourcePrimitiveArc)               #_IMA_MINIMALS_SOURCE_PRIMITIVE_ARC
    addMutantsInfo.append(numMinimalInTargetPrimitiveArc)               #_IMA_MINIMALS_TARGET_PRIMITIVE_ARC
    addMutantsInfo.append(numMinimalInPrimitiveArcOrWithSourceOrTargetInPrimitive)                  #_IMA_MINIMALS_NODE_SOURCE_OR_TARGET_PRIMITIVE
    addMutantsInfo.append(numNonMinimalInPrimitiveArcOrWithSourceOrTargetInPrimitive)                  #_IMA_NON_MINIMALS_NODE_SOURCE_OR_TARGET_PRIMITIVE
    addMutantsInfo.append(numEquivalentInPrimitiveArc)                  #_IMA_EQUIVALENTS_PRIMITIVE_ARC
    addMutantsInfo.append(numEquivalentInSourcePrimitiveArc)            #_IMA_EQUIVALENTS_SOURCE_PRIMITIVE_ARC
    addMutantsInfo.append(numEquivalentInTargetPrimitiveArc)            #_IMA_EQUIVALENTS_TARGET_PRIMITIVE_ARC
    addMutantsInfo.append(numEquivalentInPrimitiveArcOrWithSourceOrTargetInPrimitive)       #_IMA_EQUIVALENTS_NODE_SOURCE_OR_TARGET_PRIMITIVE
    addMutantsInfo.append(numNonEquivalentInPrimitiveArcOrWithSourceOrTargetInPrimitive)       #_IMA_NON_EQUIVALENTS_NODE_SOURCE_OR_TARGET_PRIMITIVE
    addMutantsInfo.append(numMinimalWithSourceNodeInPrimitiveArc)       #_IMA_MINIMALS_SOURCES_ARE_PRIMITIVE_ARC
    addMutantsInfo.append(numMinimalWithTargetNodeInPrimitiveArc)       #_IMA_MINIMALS_TARGETS_ARE_PRIMITIVE_ARC
    addMutantsInfo.append(numEquivalentWithSourceNodeInPrimitiveArc)    #_IMA_EQUIVALENTS_SOURCES_ARE_PRIMITIVE_ARC
    addMutantsInfo.append(numEquivalentWithTargetNodeInPrimitiveArc)    #_IMA_EQUIVALENTS_TARGETS_ARE_PRIMITIVE_ARC

    return addMutantsInfo

def computeEssencialInfo(mutantsInfo, minimal_Equivalent):
    #minimal_Equivalent = 0 - Minimal
    #minimal_Equivalent = 1 - Equivalent

    essential = []
    for iCount in range(len(mutantsInfo)):
        essentialRow = []
        mutantRow = mutantsInfo[iCount]
        for jCount in range(len(mutantRow)):
            # Ignora algumas colunas que não são relevantes para ML e joga a coluna minimal ou equivalente para o final
            if jCount != _IM_NUMBER and jCount != _IM_RESULT and jCount != _IM_STATUS \
                and jCount != _IM_DISTANCE_BEGIN and jCount != _IM_DISTANCE_END \
                and jCount != _IM_SOURCE_NODE and jCount != _IM_TARGET_NODE \
                and jCount != _IM_SOURCE_NODE_PRIMITIVE and jCount != _IM_TARGET_NODE_PRIMITIVE \
                and jCount != _IM_MINIMAL and jCount != _IM_EQUIVALENT:
                mutantRowInfo = mutantRow[jCount]
                essentialRow.append(mutantRowInfo)

        #_IM_OPERATOR, _IM_PROGRAM_GRAPH_NODE, _IM_PRIMITIVE_ARC, _IM_SOURCE_PRIMITIVE_ARC, _IM_TARGET_PRIMITIVE_ARC, _IM_DISTANCE_BEGIN_MIN, _IM_DISTANCE_BEGIN_MAX, _IM_DISTANCE_BEGIN_AVG, _IM_DISTANCE_END_MIN, _IM_DISTANCE_END_MAX, _IM_DISTANCE_END_AVG, _IM_EQUIVALENT, _IM_MINIMAL
        #_IM_OPERATOR, _IM_PROGRAM_GRAPH_NODE, _IM_PRIMITIVE_ARC, _IM_SOURCE_PRIMITIVE_ARC, _IM_TARGET_PRIMITIVE_ARC, _IM_DISTANCE_BEGIN_MIN, _IM_DISTANCE_BEGIN_MAX, _IM_DISTANCE_BEGIN_AVG, _IM_DISTANCE_END_MIN, _IM_DISTANCE_END_MAX, _IM_DISTANCE_END_AVG, _IM_MINIMAL, _IM_EQUIVALENT

        #Joga a coluna minimal ou equivalente para o final
        if minimal_Equivalent == 0:
            essentialRow.append(mutantRow[_IM_EQUIVALENT])
            essentialRow.append(mutantRow[_IM_MINIMAL])
        elif minimal_Equivalent == 1:
            essentialRow.append(mutantRow[_IM_MINIMAL])
            essentialRow.append(mutantRow[_IM_EQUIVALENT])
        essential.append(essentialRow)

    return essential

def formatSummaryResults(addMutantsInfo):   
    content = "Informação;Número;Total;Percentual"
    content = "{}\nMutantes minimais;{};{};{:.2f}".format(content,                          addMutantsInfo[_IMA_MINIMALS],              addMutantsInfo[_IMA_MUTANTS], addMutantsInfo[_IMA_MINIMALS] / addMutantsInfo[_IMA_MUTANTS] * 100)
    content = "{}\nMutantes equivalentes;{};{};{:.2f}".format(content,                      addMutantsInfo[_IMA_EQUIVALENTS],           addMutantsInfo[_IMA_MUTANTS], addMutantsInfo[_IMA_EQUIVALENTS] / addMutantsInfo[_IMA_MUTANTS] * 100)
    content = "{}\nMutantes nos arcos primitivos;{};{};{:.2f}".format(content,              addMutantsInfo[_IMA_PRIMITIVE_ARC],         addMutantsInfo[_IMA_MUTANTS], addMutantsInfo[_IMA_PRIMITIVE_ARC] / addMutantsInfo[_IMA_MUTANTS] * 100)
    content = "{}\nMutantes na origem dos arcos primitivos;{};{};{:.2f}".format(content,    addMutantsInfo[_IMA_SOURCE_PRIMITIVE_ARC],  addMutantsInfo[_IMA_MUTANTS], addMutantsInfo[_IMA_SOURCE_PRIMITIVE_ARC] / addMutantsInfo[_IMA_MUTANTS] * 100)
    content = "{}\nMutantes no destino dos arcos primitivos;{};{};{:.2f}".format(content,   addMutantsInfo[_IMA_TARGET_PRIMITIVE_ARC],  addMutantsInfo[_IMA_MUTANTS], addMutantsInfo[_IMA_TARGET_PRIMITIVE_ARC] / addMutantsInfo[_IMA_MUTANTS] * 100)

    content = "{}\n\nMinimais em arcos primitivos;{};{};{:.2f}".format(content,                 addMutantsInfo[_IMA_MINIMALS_PRIMITIVE_ARC],            addMutantsInfo[_IMA_MINIMALS],      addMutantsInfo[_IMA_MINIMALS_PRIMITIVE_ARC] / addMutantsInfo[_IMA_MINIMALS] * 100)
    content = "{}\nMinimais na origem dos arcos primitivos;{};{};{:.2f}".format(content,        addMutantsInfo[_IMA_MINIMALS_SOURCE_PRIMITIVE_ARC],     addMutantsInfo[_IMA_MINIMALS],      addMutantsInfo[_IMA_MINIMALS_SOURCE_PRIMITIVE_ARC] / addMutantsInfo[_IMA_MINIMALS] * 100)
    content = "{}\nMinimais no destino dos arcos primitivos;{};{};{:.2f}".format(content,       addMutantsInfo[_IMA_MINIMALS_TARGET_PRIMITIVE_ARC],     addMutantsInfo[_IMA_MINIMALS],      addMutantsInfo[_IMA_MINIMALS_TARGET_PRIMITIVE_ARC] / addMutantsInfo[_IMA_MINIMALS] * 100)
    content = "{}\nMinimais em primitivos ou com origem/destino em primitivos;{};{};{:.2f}".format(content,       addMutantsInfo[_IMA_MINIMALS_NODE_SOURCE_OR_TARGET_PRIMITIVE],     addMutantsInfo[_IMA_MINIMALS],      addMutantsInfo[_IMA_MINIMALS_NODE_SOURCE_OR_TARGET_PRIMITIVE] / addMutantsInfo[_IMA_MINIMALS] * 100)
    content = "{}\nNão Minimais em primitivos ou com origem/destino em primitivos;{};{};{:.2f}".format(content,       addMutantsInfo[_IMA_NON_MINIMALS_NODE_SOURCE_OR_TARGET_PRIMITIVE],     addMutantsInfo[_IMA_MUTANTS] - addMutantsInfo[_IMA_MINIMALS],      addMutantsInfo[_IMA_NON_MINIMALS_NODE_SOURCE_OR_TARGET_PRIMITIVE] / (addMutantsInfo[_IMA_MUTANTS] - addMutantsInfo[_IMA_MINIMALS]) * 100)
    
    content = "{}\nMinimais com nó origem em arcos primitivos;{};{};{:.2f}".format(content,       addMutantsInfo[_IMA_MINIMALS_SOURCES_ARE_PRIMITIVE_ARC],     addMutantsInfo[_IMA_MINIMALS],      addMutantsInfo[_IMA_MINIMALS_SOURCES_ARE_PRIMITIVE_ARC] / addMutantsInfo[_IMA_MINIMALS] * 100)
    content = "{}\nMinimais com nó destino em arcos primitivos;{};{};{:.2f}".format(content,       addMutantsInfo[_IMA_MINIMALS_TARGETS_ARE_PRIMITIVE_ARC],     addMutantsInfo[_IMA_MINIMALS],      addMutantsInfo[_IMA_MINIMALS_TARGETS_ARE_PRIMITIVE_ARC] / addMutantsInfo[_IMA_MINIMALS] * 100)
    
    content = "{}\nEquivalentes em arcos primitivos;{};{};{:.2f}".format(content,               addMutantsInfo[_IMA_EQUIVALENTS_PRIMITIVE_ARC],         addMutantsInfo[_IMA_EQUIVALENTS],   addMutantsInfo[_IMA_EQUIVALENTS_PRIMITIVE_ARC] / addMutantsInfo[_IMA_EQUIVALENTS] * 100)
    content = "{}\nEquivalentes na origem dos arcos primitivos;{};{};{:.2f}".format(content,    addMutantsInfo[_IMA_EQUIVALENTS_SOURCE_PRIMITIVE_ARC],  addMutantsInfo[_IMA_EQUIVALENTS],   addMutantsInfo[_IMA_EQUIVALENTS_SOURCE_PRIMITIVE_ARC] / addMutantsInfo[_IMA_EQUIVALENTS] * 100)
    content = "{}\nEquivalentes no destino dos arcos primitivos;{};{};{:.2f}".format(content,   addMutantsInfo[_IMA_EQUIVALENTS_TARGET_PRIMITIVE_ARC],  addMutantsInfo[_IMA_EQUIVALENTS],   addMutantsInfo[_IMA_EQUIVALENTS_TARGET_PRIMITIVE_ARC] / addMutantsInfo[_IMA_EQUIVALENTS] * 100)
    content = "{}\nEquivalentes em primitivos ou com origem/destino em primitivos;{};{};{:.2f}".format(content,   addMutantsInfo[_IMA_EQUIVALENTS_NODE_SOURCE_OR_TARGET_PRIMITIVE],  addMutantsInfo[_IMA_EQUIVALENTS],   addMutantsInfo[_IMA_EQUIVALENTS_NODE_SOURCE_OR_TARGET_PRIMITIVE] / addMutantsInfo[_IMA_EQUIVALENTS] * 100)
    content = "{}\nNão equivalentes em primitivos ou com origem/destino em primitivos;{};{};{:.2f}".format(content,   addMutantsInfo[_IMA_NON_EQUIVALENTS_NODE_SOURCE_OR_TARGET_PRIMITIVE],  addMutantsInfo[_IMA_MUTANTS] - addMutantsInfo[_IMA_EQUIVALENTS],   addMutantsInfo[_IMA_NON_EQUIVALENTS_NODE_SOURCE_OR_TARGET_PRIMITIVE] / (addMutantsInfo[_IMA_MUTANTS] - addMutantsInfo[_IMA_EQUIVALENTS]) * 100)
    
    content = "{}\nEquivalentes com nó origem em arcos primitivos;{};{};{:.2f}".format(content,         addMutantsInfo[_IMA_EQUIVALENTS_SOURCES_ARE_PRIMITIVE_ARC],     addMutantsInfo[_IMA_EQUIVALENTS],      addMutantsInfo[_IMA_EQUIVALENTS_SOURCES_ARE_PRIMITIVE_ARC] / addMutantsInfo[_IMA_EQUIVALENTS] * 100)
    content = "{}\nEquivalentes com nó destino em arcos primitivos;{};{};{:.2f}".format(content,        addMutantsInfo[_IMA_EQUIVALENTS_TARGETS_ARE_PRIMITIVE_ARC],     addMutantsInfo[_IMA_EQUIVALENTS],      addMutantsInfo[_IMA_EQUIVALENTS_TARGETS_ARE_PRIMITIVE_ARC] / addMutantsInfo[_IMA_EQUIVALENTS] * 100)

    return content

def main(_baseExperimentFolder, _baseFolder, executionMode):
    #executionMode |    1 - Run and analyze
    #                   2 - Run
    #                   3 - Analyze

    if int(executionMode) >= 1 and int(executionMode) <=3 :
        print ('####################################################################')
        print ('#\t   Executing script to find minimal mutants properties\t   #')
        print ('#\t\t      ' + util.formatNow() + '\t\t\t   #')
        print ('####################################################################')

        ####################
        # Set main variables
        ####################
        baseExperimentFolder = _baseExperimentFolder
        baseFolder = _baseFolder
        sourceFile = baseFolder[baseFolder.rfind("/") + 1:]
        sessionName = sourceFile
        executableFile = sessionName
        executionType = "research"
        directory = baseFolder
        unitName = util.getContentFromFile("{baseFolder}/unit.txt".format(baseFolder = baseFolder))

        if int(executionMode) == 1 or int(executionMode) == 2:
            #################
            # Execute proteum
            #################
            executeProteum(baseFolder, sourceFile, sessionName, executableFile, executionType, directory, unitName)
        
        if int(executionMode) == 1 or int(executionMode) == 3:
            #####################
            # Get minimal mutants
            #####################
            print ('\n##### \tBuscando mutantes minimais ' + util.formatNow() + '\t#####'         )
            minimalMutants = getMinimalMutants(baseFolder, sourceFile)

            ######################
            # Simplifying GFC file
            ######################
            # Desabilitado pois estou utilizando GFC do Pedro
            #print ('\n##### \tSimplificando arquivo GFC ' + util.formatNow() + '\t#####'         )
            #prot2PokeMain("{baseFolder}/__{sourceFile}.gfc".format(
                #baseFolder = baseFolder, sourceFile = sourceFile))

            ################################
            # Get basic mutants informations
            ################################
            print ('\n##### \tBuscando e calculando informações dos mutantes ' + util.formatNow() + '\t#####')
            mutantsHeader, mutantsInfo = getMutantsInfo(baseFolder, minimalMutants, sessionName, unitName)

            ################################################
            # Write csv File with basic mutants informations
            ################################################
            print ('\n##### \tGerando arquivo com informações dos mutantes ' + util.formatNow() + '\t#####')
            fileNameResults = "{baseFolder}/log/{unitName}_result.csv".format(
                unitName = unitName, baseFolder = baseFolder)
            util.writeInCsvFile(fileNameResults, mutantsInfo, mutantsHeader)

            #####################################
            # Get additional mutants informations
            #####################################
            addMutantsInfo = computeAdicionalMutantsInfo(mutantsInfo)
            fileNameSummaryResults = "{baseFolder}/log/{unitName}_summary_results.csv".format(
                unitName = unitName, baseFolder = baseFolder)
            util.write(fileNameSummaryResults, formatSummaryResults(addMutantsInfo))

            ###########################################################
            # Write mutants info to compute machine learning algorithms
            ###########################################################
            essentialInfo = computeEssencialInfo(mutantsInfo, minimal_Equivalent=1)
            essentialFileName = "{}/mutants.csv".format(baseExperimentFolder)                   # Gera apenas um arquivo com todos os mutantes
            #essentialFileName = "{}/{}_mutants.csv".format(baseExperimentFolder, sessionName)  # Gera um arquivo para cada programa com todos os seus mutantes
            util.writeInCsvFile(essentialFileName, essentialInfo, mode="a+")


if __name__ == '__main__':
    if len(sys.argv) > 3:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print ('##### Exit #####')