#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys

from Others.ComputeMinimalMuta import computeMinimal
from Others.prot2poke import prot2PokeMain

import proteum
import util
import gfcUtils

import constants

def getMinimalMutants(baseFolder, sourceFile):
    fileName = "{}/log/minimal.txt".format(baseFolder)
    #fileName = "{}/arc_prim/minimal.txt".format(baseFolder) #Utilizar as informações dos minimais já passadas

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
            minimalMutants.append(int(line.strip()))

    return minimalMutants     

'''
    Função principal que executa todo o passo a passo na Proteum para geração dos mutantes
'''
def executeProteum(baseFolder, sourceFile, sessionName, executableFile, executionType, directory, units):
    if not util.pathExists("{}/log".format(baseFolder)):
        util.createFolder("{}/log".format(baseFolder))
    
    #####################
    # Create Test Session
    #####################
    print ('\n##### \tCriando sessão de testes ' + util.formatNow() + '\t#####'    )
    
    # Inclui o arquivo driver.c na compilação caso ele exista na pasta
    driver = 'driver.c' if util.pathExists('{}/driver.c'.format(baseFolder)) else ''
    proteum.createSession(baseFolder, sessionName, executionType, sourceFile, executableFile, directory, driver)

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
    for unitName in str(units).splitlines():
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

def getMutantInfosFromText(mutant):
    mutant = mutant.strip()
    start = mutant.find('\n')

    if start == -1:
        return None, None, None, None, None, None, None, None

    # Define o índice inicial de cada informação
    start_Number = start
    start_Status = mutant.find('Status ') + len('Status ')
    start_Operator = mutant.find('Operator: ') + len('Operator: ')
    start_Operator = mutant.find('(', start_Operator) + 1
    start_ProgramGraphNode = mutant.find('Program graph node: ') + len('Program graph node: ')
    start_OffSet = mutant.find('Offset: ') + len('Offset: ')
    start_GetOut = mutant.find('get out ', start_OffSet) + len('get out ')
    start_DescriptorSize = mutant.find('Descriptor size.: ') + len('Descriptor size.: ')
    start_CallingFunction = mutant.find('Calling function starts at: ') + len('Calling function starts at: ')

    # Define o índice final de cada informação
    end_Number = mutant.find('\n', start_Number)
    end_Status = mutant.find('\n', start_Status)
    end_Operator = mutant.find(')', start_Operator)
    end_ProgramGraphNode = mutant.find('\n', start_ProgramGraphNode)
    end_Offset = mutant.find(',', start_OffSet)
    end_GetOut = mutant.find(' characters', start_GetOut)
    end_DescriptorSize = mutant.find('\n', start_DescriptorSize)
    end_CallingFunction = mutant.find('\n', start_CallingFunction)

    # Encontra as informaçõões entre os índices inicial e final
    mutantNumber = int(str(mutant[0 : end_Number ]).strip())
    status = str(mutant[start_Status : end_Status ]).strip()
    operator = str(mutant[start_Operator : end_Operator ]).strip()
    programGraphNode = str(mutant[start_ProgramGraphNode : end_ProgramGraphNode ]).strip()
    offSet = int(str(mutant[start_OffSet : end_Offset ]).strip())
    getOut = int(str(mutant[start_GetOut : end_GetOut ]).strip())
    descriptorSize = int(str(mutant[start_DescriptorSize : end_DescriptorSize ]).strip())
    callingFunction = int(str(mutant[start_CallingFunction : end_CallingFunction ]).strip())

    return mutantNumber, status, operator, programGraphNode, offSet, getOut, descriptorSize, callingFunction

def getMutantsInfo(baseFolder, minimalMutants, sessionName, units):
    arrMutantsInfo = []
    arrHeaderMutants = []
    
    mutantsInfoFileName = "{baseFolder}/log/mutants.txt".format(baseFolder = baseFolder)
    if (not util.pathExists(mutantsInfoFileName)):
        proteum.showMutants(sessionName, baseFolder, "")

    mutantsInfoFile = util.getContentFromFile(mutantsInfoFileName)

    # Código que foi utilizado para gerar o mutante
    codeFile = '{baseFolder}/__{sessionName}.c'.format(baseFolder = baseFolder, sessionName = sessionName)

    # Coleta as informações para cada uma das funções executadas
    for unitName in units.splitlines():
        gfcFileName = "{baseFolder}/arc_prim/{unitName}.gfc".format(baseFolder = baseFolder, unitName = unitName) #Utiliza as informações já passadas
        arcPrimFileName = "{baseFolder}/arc_prim/{unitName}.tes".format(baseFolder = baseFolder, unitName = unitName)
        gfc, numNodes = gfcUtils.gfcMain(gfcFileName, arcPrimFileName)

        # Propriedade responsável por contar o número de mutantes em cada nó do GFC
        mutantsOnNodes = []

        # Divide o arquivo de mutantes pelo caracter # (cada um representa um mutante)
        mutants = mutantsInfoFile.split("#")
        for mutant in mutants:
            
            # Variável utilizada para identificar o último mutante analisado
            lastICount = 0

            # Busca todas as informações relevantes do mutante que estão no arquivo
            mutantNumber, status, operator, programGraphNode, offSet, getOut, descriptorSize, callingFunction = getMutantInfosFromText(mutant)
            if mutantNumber == None:
                continue

            # Caso a função onde ocorreu a mutação for diferente da função analisada, ignora este mutante pois ele será analisado em outro momento          
            functionName = str(gfcUtils.getOffsetFromCode(codeFile, callingFunction, descriptorSize))
            functionName = functionName[2: functionName.find('(', 2)].strip()
            if functionName != unitName:
                continue

            # Define se o mutante é minimal ou não
            minimal = mutantNumber in minimalMutants

            # Define se o mutante é equivalente ou não
            equivalent = status.__contains__("Equivalent")

            # Verifica o tipo de declaração que houve mutação
            descriptor, descriptor_line = gfcUtils.getOffsetFromCode(codeFile, offSet, getOut)
            typeStatement = gfcUtils.getTypeStatementFromCode(descriptor, descriptor_line, sessionName)

            # Calcula o número de mutantes no nó da mutação
            mutantsOnNodes = gfcUtils.getMutantsOnNode(mutantsOnNodes, programGraphNode)

            # Busca os nós origens e destinos
                # Calcula as informações de distância do nó da mutação até os nós inicias \
                    # Calcula as informações de distância do nó da mutação até os nós finais \
                        # Calcula as informações de pertencimento ou não aos arcos primitivos
            sourceNode, targetNode, \
                distanceBegin, distanceBegin_min, distanceBegin_max, distanceBegin_avg,\
                    distanceEnd, distanceEnd_min, distanceEnd_max, distanceEnd_avg,\
                        sourcePrimitiveArc, targetPrimitiveArc, sourcesNodeIsPrimitive, targetsNodeIsPrimitive = gfcUtils.getInfoNode(gfc, programGraphNode, numNodes)

            # Define se o mutante pertence aos arcos primitivos ou não
            primitiveArc = sourcePrimitiveArc or targetPrimitiveArc

            # Reune todas as informações dos mutantes num array
            arrMutantInfo = []
            arrMutantInfo.append(mutantNumber)
            arrMutantInfo.append(status) # Temporariamente a coluna _IM_RESULT vai conter o status
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
            arrMutantInfo.append('[MutantsOnNode]')
            arrMutantInfo.append(typeStatement)

            # Adiciona as informações do mutante num outro array que contém todas as informações de todos os mutantes
            arrMutantsInfo.append(arrMutantInfo)

        # Atualiza as informações sobre os nós dos mutantes
        for iCount in range(lastICount, len(arrMutantsInfo), 1):
            arrMutantsInfo[iCount][constants._IM_COMPLEXITY] = gfcUtils.getNumMutantsOnNode(mutantsOnNodes, arrMutantsInfo[iCount][constants._IM_PROGRAM_GRAPH_NODE])
            lastICount = iCount

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
    arrHeaderMutants.append("Complexity")
    arrHeaderMutants.append("Type Statement")

    return arrHeaderMutants, arrMutantsInfo

def computeAdicionalMutantsInfo(mutantsInfo):

    addMutantsInfo = []

    #Basic Info
    totalMinimal = list(filter(lambda info: info[constants._IM_MINIMAL] == '1', mutantsInfo))
    #totalNonMinimal = list(filter(lambda info: info[_IM_MINIMAL] == '0', mutantsInfo))
    totalEquivalents = list(filter(lambda info: info[constants._IM_EQUIVALENT] == '1', mutantsInfo))
    #totalNonEquivalents = list(filter(lambda info: info[_IM_EQUIVALENT] == '0', mutantsInfo))
    totalPrimitiveArc = list(filter(lambda info: info[constants._IM_PRIMITIVE_ARC] == '1', mutantsInfo))
    totalSourcePrimitiveArc = list(filter(lambda info: info[constants._IM_SOURCE_PRIMITIVE_ARC] == '1', mutantsInfo))
    totalTargetPrimitiveArc = list(filter(lambda info: info[constants._IM_TARGET_PRIMITIVE_ARC] == '1', mutantsInfo))

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
    minimalInPrimitiveArc = list(filter(lambda info: info[constants._IM_PRIMITIVE_ARC] == '1', totalMinimal))
    minimalInSourcePrimitiveArc = list(filter(lambda info: info[constants._IM_SOURCE_PRIMITIVE_ARC] == '1', totalMinimal))
    minimalInTargetPrimitiveArc = list(filter(lambda info: info[constants._IM_TARGET_PRIMITIVE_ARC] == '1', totalMinimal))
    equivalentInPrimitiveArc = list(filter(lambda info: info[constants._IM_PRIMITIVE_ARC] == '1', totalEquivalents))
    equivalentInSourcePrimitiveArc = list(filter(lambda info: info[constants._IM_SOURCE_PRIMITIVE_ARC] == '1', totalEquivalents))
    equivalentInTargetPrimitiveArc = list(filter(lambda info: info[constants._IM_TARGET_PRIMITIVE_ARC] == '1', totalEquivalents))
    
    #minimalInPrimitiveArcOrWithSourceOrTargetInPrimitive = list(filter(lambda info:
    #    info[_IM_PRIMITIVE_ARC] == '1' or
    #    list(info[_IM_SOURCE_NODE_PRIMITIVE]).__contains__(1) or 
    #    list(info[_IM_TARGET_NODE_PRIMITIVE]).__contains__(1),
    #totalMinimal))
    #nonMinimalInPrimitiveArcOrWithSourceOrTargetInPrimitive = list(filter(lambda info:
    #    info[_IM_PRIMITIVE_ARC] == '1' or
    #    list(info[_IM_SOURCE_NODE_PRIMITIVE]).__contains__(1) or 
    #    list(info[_IM_TARGET_NODE_PRIMITIVE]).__contains__(1),
    #totalNonMinimal))
    #minimalWithSourceNodeInPrimitiveArc = list(filter(lambda info: list(info[_IM_SOURCE_NODE_PRIMITIVE]).__contains__(1), totalMinimal))
    #minimalWithTargetNodeInPrimitiveArc = list(filter(lambda info: list(info[_IM_TARGET_NODE_PRIMITIVE]).__contains__(1), totalMinimal))
    
    #equivalentInPrimitiveArcOrWithSourceOrTargetInPrimitive = list(filter(lambda info:
        #info[_IM_PRIMITIVE_ARC] == '1' or
        #list(info[_IM_SOURCE_NODE_PRIMITIVE]).__contains__(1) or 
        #list(info[_IM_TARGET_NODE_PRIMITIVE]).__contains__(1),
    #totalEquivalents))
    #nonEquivalentInPrimitiveArcOrWithSourceOrTargetInPrimitive = list(filter(lambda info:
        #info[_IM_PRIMITIVE_ARC] == '1' or
        #list(info[_IM_SOURCE_NODE_PRIMITIVE]).__contains__(1) or 
        #list(info[_IM_TARGET_NODE_PRIMITIVE]).__contains__(1),
    #totalNonEquivalents))    
    #equivalentWithSourceNodeInPrimitiveArc = list(filter(lambda info: list(info[_IM_SOURCE_NODE_PRIMITIVE]).__contains__(1), totalEquivalents))
    #equivalentWithTargetNodeInPrimitiveArc = list(filter(lambda info: list(info[_IM_TARGET_NODE_PRIMITIVE]).__contains__(1), totalEquivalents))

    numMinimalInPrimitiveArc = len(minimalInPrimitiveArc)
    numMinimalInSourcePrimitiveArc = len(minimalInSourcePrimitiveArc)
    numMinimalInTargetPrimitiveArc = len(minimalInTargetPrimitiveArc)
    numEquivalentInPrimitiveArc = len(equivalentInPrimitiveArc)
    numEquivalentInSourcePrimitiveArc = len(equivalentInSourcePrimitiveArc)
    numEquivalentInTargetPrimitiveArc = len(equivalentInTargetPrimitiveArc)

    #numMinimalInPrimitiveArcOrWithSourceOrTargetInPrimitive = len(minimalInPrimitiveArcOrWithSourceOrTargetInPrimitive)
    #numNonMinimalInPrimitiveArcOrWithSourceOrTargetInPrimitive = len(nonMinimalInPrimitiveArcOrWithSourceOrTargetInPrimitive)
    #numMinimalWithSourceNodeInPrimitiveArc = len(minimalWithSourceNodeInPrimitiveArc)
    #numMinimalWithTargetNodeInPrimitiveArc = len(minimalWithTargetNodeInPrimitiveArc)

    #numEquivalentInPrimitiveArcOrWithSourceOrTargetInPrimitive = len(equivalentInPrimitiveArcOrWithSourceOrTargetInPrimitive)
    #numNonEquivalentInPrimitiveArcOrWithSourceOrTargetInPrimitive = len(nonEquivalentInPrimitiveArcOrWithSourceOrTargetInPrimitive)
    #numEquivalentWithSourceNodeInPrimitiveArc = len(equivalentWithSourceNodeInPrimitiveArc)
    #numEquivalentWithTargetNodeInPrimitiveArc = len(equivalentWithTargetNodeInPrimitiveArc)
                        

    addMutantsInfo.append(numMinimalInPrimitiveArc)                     #_IMA_MINIMALS_PRIMITIVE_ARC
    addMutantsInfo.append(numMinimalInSourcePrimitiveArc)               #_IMA_MINIMALS_SOURCE_PRIMITIVE_ARC
    addMutantsInfo.append(numMinimalInTargetPrimitiveArc)               #_IMA_MINIMALS_TARGET_PRIMITIVE_ARC
    #addMutantsInfo.append(numMinimalInPrimitiveArcOrWithSourceOrTargetInPrimitive)                  #_IMA_MINIMALS_NODE_SOURCE_OR_TARGET_PRIMITIVE
    #addMutantsInfo.append(numNonMinimalInPrimitiveArcOrWithSourceOrTargetInPrimitive)                  #_IMA_NON_MINIMALS_NODE_SOURCE_OR_TARGET_PRIMITIVE
    addMutantsInfo.append(numEquivalentInPrimitiveArc)                  #_IMA_EQUIVALENTS_PRIMITIVE_ARC
    addMutantsInfo.append(numEquivalentInSourcePrimitiveArc)            #_IMA_EQUIVALENTS_SOURCE_PRIMITIVE_ARC
    addMutantsInfo.append(numEquivalentInTargetPrimitiveArc)            #_IMA_EQUIVALENTS_TARGET_PRIMITIVE_ARC
    #addMutantsInfo.append(numEquivalentInPrimitiveArcOrWithSourceOrTargetInPrimitive)       #_IMA_EQUIVALENTS_NODE_SOURCE_OR_TARGET_PRIMITIVE
    #addMutantsInfo.append(numNonEquivalentInPrimitiveArcOrWithSourceOrTargetInPrimitive)       #_IMA_NON_EQUIVALENTS_NODE_SOURCE_OR_TARGET_PRIMITIVE
    #addMutantsInfo.append(numMinimalWithSourceNodeInPrimitiveArc)       #_IMA_MINIMALS_SOURCES_ARE_PRIMITIVE_ARC
    #addMutantsInfo.append(numMinimalWithTargetNodeInPrimitiveArc)       #_IMA_MINIMALS_TARGETS_ARE_PRIMITIVE_ARC
    #addMutantsInfo.append(numEquivalentWithSourceNodeInPrimitiveArc)    #_IMA_EQUIVALENTS_SOURCES_ARE_PRIMITIVE_ARC
    #addMutantsInfo.append(numEquivalentWithTargetNodeInPrimitiveArc)    #_IMA_EQUIVALENTS_TARGETS_ARE_PRIMITIVE_ARC

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
            if jCount != constants._IM_NUMBER and jCount != constants._IM_RESULT and jCount != constants._IM_STATUS \
                and jCount != constants._IM_DISTANCE_BEGIN and jCount !=constants. _IM_DISTANCE_END \
                and jCount != constants._IM_SOURCE_NODE and jCount != constants._IM_TARGET_NODE \
                and jCount != constants._IM_SOURCE_NODE_PRIMITIVE and jCount != constants._IM_TARGET_NODE_PRIMITIVE \
                and jCount != constants._IM_MINIMAL and jCount != constants._IM_EQUIVALENT:
                mutantRowInfo = mutantRow[jCount]
                essentialRow.append(mutantRowInfo)

        #_IM_OPERATOR, _IM_PROGRAM_GRAPH_NODE, _IM_PRIMITIVE_ARC, _IM_SOURCE_PRIMITIVE_ARC, _IM_TARGET_PRIMITIVE_ARC, _IM_DISTANCE_BEGIN_MIN, _IM_DISTANCE_BEGIN_MAX, _IM_DISTANCE_BEGIN_AVG, _IM_DISTANCE_END_MIN, _IM_DISTANCE_END_MAX, _IM_DISTANCE_END_AVG, _IM_COMPLEXITY, _IM_EQUIVALENT, _IM_MINIMAL
        #_IM_OPERATOR, _IM_PROGRAM_GRAPH_NODE, _IM_PRIMITIVE_ARC, _IM_SOURCE_PRIMITIVE_ARC, _IM_TARGET_PRIMITIVE_ARC, _IM_DISTANCE_BEGIN_MIN, _IM_DISTANCE_BEGIN_MAX, _IM_DISTANCE_BEGIN_AVG, _IM_DISTANCE_END_MIN, _IM_DISTANCE_END_MAX, _IM_DISTANCE_END_AVG, _IM_COMPLEXITY, _IM_MINIMAL, _IM_EQUIVALENT

        #Joga a coluna minimal ou equivalente para o final
        if minimal_Equivalent == 0:
            essentialRow.append(mutantRow[constants._IM_EQUIVALENT])
            essentialRow.append(mutantRow[constants._IM_MINIMAL])
        elif minimal_Equivalent == 1:
            essentialRow.append(mutantRow[constants._IM_MINIMAL])
            essentialRow.append(mutantRow[constants._IM_EQUIVALENT])
        essential.append(essentialRow)

    return essential

def formatSummaryResults(addMutantsInfo):   
    content = "Informação;Número;Total;Percentual"
    content = "{}\nMutantes minimais;{};{};{:.2f}".format(content,                          addMutantsInfo[constants._IMA_MINIMALS],              addMutantsInfo[constants._IMA_MUTANTS], addMutantsInfo[constants._IMA_MINIMALS] / addMutantsInfo[constants._IMA_MUTANTS] * 100)
    content = "{}\nMutantes equivalentes;{};{};{:.2f}".format(content,                      addMutantsInfo[constants._IMA_EQUIVALENTS],           addMutantsInfo[constants._IMA_MUTANTS], addMutantsInfo[constants._IMA_EQUIVALENTS] / addMutantsInfo[constants._IMA_MUTANTS] * 100)
    content = "{}\nMutantes nos arcos primitivos;{};{};{:.2f}".format(content,              addMutantsInfo[constants._IMA_PRIMITIVE_ARC],         addMutantsInfo[constants._IMA_MUTANTS], addMutantsInfo[constants._IMA_PRIMITIVE_ARC] / addMutantsInfo[constants._IMA_MUTANTS] * 100)
    content = "{}\nMutantes na origem dos arcos primitivos;{};{};{:.2f}".format(content,    addMutantsInfo[constants._IMA_SOURCE_PRIMITIVE_ARC],  addMutantsInfo[constants._IMA_MUTANTS], addMutantsInfo[constants._IMA_SOURCE_PRIMITIVE_ARC] / addMutantsInfo[constants._IMA_MUTANTS] * 100)
    content = "{}\nMutantes no destino dos arcos primitivos;{};{};{:.2f}".format(content,   addMutantsInfo[constants._IMA_TARGET_PRIMITIVE_ARC],  addMutantsInfo[constants._IMA_MUTANTS], addMutantsInfo[constants._IMA_TARGET_PRIMITIVE_ARC] / addMutantsInfo[constants._IMA_MUTANTS] * 100)

    content = "{}\n\nMinimais em arcos primitivos;{};{};{:.2f}".format(content,                 addMutantsInfo[constants._IMA_MINIMALS_PRIMITIVE_ARC],            addMutantsInfo[constants._IMA_MINIMALS],      addMutantsInfo[constants._IMA_MINIMALS_PRIMITIVE_ARC] / addMutantsInfo[constants._IMA_MINIMALS] * 100)
    content = "{}\nMinimais na origem dos arcos primitivos;{};{};{:.2f}".format(content,        addMutantsInfo[constants._IMA_MINIMALS_SOURCE_PRIMITIVE_ARC],     addMutantsInfo[constants._IMA_MINIMALS],      addMutantsInfo[constants._IMA_MINIMALS_SOURCE_PRIMITIVE_ARC] / addMutantsInfo[constants._IMA_MINIMALS] * 100)
    content = "{}\nMinimais no destino dos arcos primitivos;{};{};{:.2f}".format(content,       addMutantsInfo[constants._IMA_MINIMALS_TARGET_PRIMITIVE_ARC],     addMutantsInfo[constants._IMA_MINIMALS],      addMutantsInfo[constants._IMA_MINIMALS_TARGET_PRIMITIVE_ARC] / addMutantsInfo[constants._IMA_MINIMALS] * 100)
    #content = "{}\nMinimais em primitivos ou com origem/destino em primitivos;{};{};{:.2f}".format(content,       addMutantsInfo[_IMA_MINIMALS_NODE_SOURCE_OR_TARGET_PRIMITIVE],     addMutantsInfo[_IMA_MINIMALS],      addMutantsInfo[_IMA_MINIMALS_NODE_SOURCE_OR_TARGET_PRIMITIVE] / addMutantsInfo[_IMA_MINIMALS] * 100)
    #content = "{}\nNão Minimais em primitivos ou com origem/destino em primitivos;{};{};{:.2f}".format(content,       addMutantsInfo[_IMA_NON_MINIMALS_NODE_SOURCE_OR_TARGET_PRIMITIVE],     addMutantsInfo[_IMA_MUTANTS] - addMutantsInfo[_IMA_MINIMALS],      addMutantsInfo[_IMA_NON_MINIMALS_NODE_SOURCE_OR_TARGET_PRIMITIVE] / (addMutantsInfo[_IMA_MUTANTS] - addMutantsInfo[_IMA_MINIMALS]) * 100)
    
    #content = "{}\nMinimais com nó origem em arcos primitivos;{};{};{:.2f}".format(content,       addMutantsInfo[_IMA_MINIMALS_SOURCES_ARE_PRIMITIVE_ARC],     addMutantsInfo[_IMA_MINIMALS],      addMutantsInfo[_IMA_MINIMALS_SOURCES_ARE_PRIMITIVE_ARC] / addMutantsInfo[_IMA_MINIMALS] * 100)
    #content = "{}\nMinimais com nó destino em arcos primitivos;{};{};{:.2f}".format(content,       addMutantsInfo[_IMA_MINIMALS_TARGETS_ARE_PRIMITIVE_ARC],     addMutantsInfo[_IMA_MINIMALS],      addMutantsInfo[_IMA_MINIMALS_TARGETS_ARE_PRIMITIVE_ARC] / addMutantsInfo[_IMA_MINIMALS] * 100)
    
    content = "{}\nEquivalentes em arcos primitivos;{};{};{:.2f}".format(content,               addMutantsInfo[constants._IMA_EQUIVALENTS_PRIMITIVE_ARC],         addMutantsInfo[constants._IMA_EQUIVALENTS],   addMutantsInfo[constants._IMA_EQUIVALENTS_PRIMITIVE_ARC] / addMutantsInfo[constants._IMA_EQUIVALENTS] * 100)
    content = "{}\nEquivalentes na origem dos arcos primitivos;{};{};{:.2f}".format(content,    addMutantsInfo[constants._IMA_EQUIVALENTS_SOURCE_PRIMITIVE_ARC],  addMutantsInfo[constants._IMA_EQUIVALENTS],   addMutantsInfo[constants._IMA_EQUIVALENTS_SOURCE_PRIMITIVE_ARC] / addMutantsInfo[constants._IMA_EQUIVALENTS] * 100)
    content = "{}\nEquivalentes no destino dos arcos primitivos;{};{};{:.2f}".format(content,   addMutantsInfo[constants._IMA_EQUIVALENTS_TARGET_PRIMITIVE_ARC],  addMutantsInfo[constants._IMA_EQUIVALENTS],   addMutantsInfo[constants._IMA_EQUIVALENTS_TARGET_PRIMITIVE_ARC] / addMutantsInfo[constants._IMA_EQUIVALENTS] * 100)
    #content = "{}\nEquivalentes em primitivos ou com origem/destino em primitivos;{};{};{:.2f}".format(content,   addMutantsInfo[_IMA_EQUIVALENTS_NODE_SOURCE_OR_TARGET_PRIMITIVE],  addMutantsInfo[_IMA_EQUIVALENTS],   addMutantsInfo[_IMA_EQUIVALENTS_NODE_SOURCE_OR_TARGET_PRIMITIVE] / addMutantsInfo[_IMA_EQUIVALENTS] * 100)
    #content = "{}\nNão equivalentes em primitivos ou com origem/destino em primitivos;{};{};{:.2f}".format(content,   addMutantsInfo[_IMA_NON_EQUIVALENTS_NODE_SOURCE_OR_TARGET_PRIMITIVE],  addMutantsInfo[_IMA_MUTANTS] - addMutantsInfo[_IMA_EQUIVALENTS],   addMutantsInfo[_IMA_NON_EQUIVALENTS_NODE_SOURCE_OR_TARGET_PRIMITIVE] / (addMutantsInfo[_IMA_MUTANTS] - addMutantsInfo[_IMA_EQUIVALENTS]) * 100)
    
    #content = "{}\nEquivalentes com nó origem em arcos primitivos;{};{};{:.2f}".format(content,         addMutantsInfo[_IMA_EQUIVALENTS_SOURCES_ARE_PRIMITIVE_ARC],     addMutantsInfo[_IMA_EQUIVALENTS],      addMutantsInfo[_IMA_EQUIVALENTS_SOURCES_ARE_PRIMITIVE_ARC] / addMutantsInfo[_IMA_EQUIVALENTS] * 100)
    #content = "{}\nEquivalentes com nó destino em arcos primitivos;{};{};{:.2f}".format(content,        addMutantsInfo[_IMA_EQUIVALENTS_TARGETS_ARE_PRIMITIVE_ARC],     addMutantsInfo[_IMA_EQUIVALENTS],      addMutantsInfo[_IMA_EQUIVALENTS_TARGETS_ARE_PRIMITIVE_ARC] / addMutantsInfo[_IMA_EQUIVALENTS] * 100)

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
        #unitName = util.getContentFromFile("{baseFolder}/unit.txt".format(baseFolder = baseFolder))
        units = util.getContentFromFile("{baseFolder}/unit.txt".format(baseFolder = baseFolder))

        if int(executionMode) == 1 or int(executionMode) == 2:
            #################
            # Execute proteum
            #################
            executeProteum(baseFolder, sourceFile, sessionName, executableFile, executionType, directory, units)
        
        if int(executionMode) == 1 or int(executionMode) == 3:
            #####################
            # Get minimal mutants
            #####################
            print ('\n##### \tBuscando mutantes minimais ' + util.formatNow() + '\t#####'         )
            minimalMutants = getMinimalMutants(baseFolder, sourceFile)

            ######################
            # Simplifying GFC file
            ######################
            # Desabilitado pois estou utilizando GFC já passados
            #print ('\n##### \tSimplificando arquivo GFC ' + util.formatNow() + '\t#####'         )
            #prot2PokeMain("{baseFolder}/__{sourceFile}.gfc".format(
                #baseFolder = baseFolder, sourceFile = sourceFile))

            ################################
            # Get basic mutants informations
            ################################
            print ('\n##### \tBuscando e calculando informações dos mutantes ' + util.formatNow() + '\t#####')
            mutantsHeader, mutantsInfo = getMutantsInfo(baseFolder, minimalMutants, sessionName, units)

            ################################################
            # Write csv File with basic mutants informations
            ################################################
            print ('\n##### \tGerando arquivo com informações dos mutantes ' + util.formatNow() + '\t#####')
            fileNameResults = "{baseFolder}/log/{sessionName}_result.csv".format(
                sessionName = sessionName, baseFolder = baseFolder)
            util.writeInCsvFile(fileNameResults, mutantsInfo, mutantsHeader)

            #####################################
            # Get additional mutants informations
            #####################################
            addMutantsInfo = computeAdicionalMutantsInfo(mutantsInfo)
            fileNameSummaryResults = "{baseFolder}/log/{sessionName}_summary_results.csv".format(
                sessionName = sessionName, baseFolder = baseFolder)
            util.write(fileNameSummaryResults, formatSummaryResults(addMutantsInfo))

            ###########################################################
            # Write mutants info to compute machine learning algorithms
            ###########################################################
            ### --- Minimals --- ###
            essentialInfo = computeEssencialInfo(mutantsInfo, minimal_Equivalent=0)
            essentialFileName = "{}/mutants_minimals.csv".format(baseExperimentFolder)          # Gera apenas um arquivo com todos os mutantes
            #essentialFileName = "{}/{}_mutants.csv".format(baseExperimentFolder, sessionName)  # Gera um arquivo para cada programa com todos os seus mutantes
            util.writeInCsvFile(essentialFileName, essentialInfo, mode="a+")

            ### --- Equivalents --- ###
            essentialInfo = computeEssencialInfo(mutantsInfo, minimal_Equivalent=1)
            essentialFileName = "{}/mutants_equivalents.csv".format(baseExperimentFolder)       # Gera apenas um arquivo com todos os mutantes
            #essentialFileName = "{}/{}_mutants.csv".format(baseExperimentFolder, sessionName)  # Gera um arquivo para cada programa com todos os seus mutantes
            util.writeInCsvFile(essentialFileName, essentialInfo, mode="a+")


if __name__ == '__main__':
    if len(sys.argv) > 3:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print ('##### Exit #####')