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

import util
import sys
import subprocess

import constants

def getGfc(fileName, arcPrimFile, showOutput):

    # Busca as informações referente aos arcos primitivos daquele programa
    primitiveNodes = util.getContentFromFile(arcPrimFile)
    
    # Busca o conteúdo do arquivo .gfc
    gfc = util.getContentFromFile(fileName).replace('\t', '')
    gfc, numNodes = prepareGFC(gfc, getPrimitiveNode(primitiveNodes))

    if showOutput:
        for node in gfc:   print('Nó: {}\tOrigens: {}\tDestinos: {}\tDistância do início: {}\tDistância do fim: {}'.format(node, gfc[node][constants._iNSources], gfc[node][constants._iNTargets], gfc[node][constants._iNDistancesBegin], gfc[node][constants._iNDistancesEnd]))

    return gfc, numNodes

'''
    Função responsável por calcular (incrementar ou adicionar) o número de mutantes em dado nó do GFC
'''
def getMutantsOnNode(mutantsOnNodes, programGraphNode):
    isOnList = False

    if len(mutantsOnNodes) > 1:
        for nodeNumber, valueNode in mutantsOnNodes:
            if programGraphNode == nodeNumber:
                isOnList = True
                indexNode = mutantsOnNodes.index([nodeNumber, valueNode])
                
                valueNode = valueNode + 1
                mutantsOnNodes[indexNode] = [nodeNumber, valueNode]
    
    if isOnList == False:
        mutantsOnNodes.append([programGraphNode, 1])

    return mutantsOnNodes

'''
    Função responsável por obter o número de mutantes em dado nó do GFC
'''
def getNumMutantsOnNode(mutantsOnNodes, programGraphNode):
    for node, value in mutantsOnNodes:
        if programGraphNode == node:
            return value

    return -1

def getOffsetFromCode(codeFile, beginOffset, getOut):
    contentFile = util.getContentFromFile(codeFile)
    
    beginLine = contentFile.rfind('\n', 0, beginOffset)

    endOffset = beginOffset + getOut
    descriptor = (contentFile[beginOffset - 1: endOffset])
    descriptor_line = (contentFile[beginLine : endOffset])

    return descriptor, descriptor_line

def getTypeStatementFromCode(descriptor, descriptor_line, sessionName):
    descriptor_line = str(descriptor_line).replace('\n', '', 1).replace('\t', '')

    typeStatement = ''
    if descriptor_line.__contains__('\n') and descriptor_line.replace('\n', '', 1).__contains__('\n'):
        typeStatement = 'Block'
    elif descriptor_line.__contains__('while'):
        typeStatement = 'While'
    elif descriptor_line.__contains__('for'):
        typeStatement = 'For'
    elif descriptor_line.__contains__('if'):
        typeStatement = 'If'
    elif descriptor_line.__contains__('=') or descriptor_line.__contains__('++') or descriptor_line.__contains__('--'):
        typeStatement = 'Assignment'
    elif descriptor_line.__contains__('return'):
        typeStatement = 'Return'
    elif descriptor_line.__contains__('('):
        typeStatement = 'Function Call'
    else:
        typeStatement = 'Declaration'

    return typeStatement

def getInfoNode(gfc, node, numNodes):
    if int(node) > int(numNodes):
        return -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
    
    source = getSource(gfc, node)
    target = getTarget(gfc, node)

    distanceFromBegin = gfc[node][constants._iNDistancesBegin]
    distanceFromBegin_min = gfc[node][constants._iNDistancesBegin_min]
    distanceFromBegin_max = gfc[node][constants._iNDistancesBegin_max]
    distanceFromBegin_avg = gfc[node][constants._iNDistancesBegin_avg]
    
    distanceFromEnd = gfc[node][constants._iNDistancesEnd]
    distanceFromEnd_min = gfc[node][constants._iNDistancesEnd_min]
    distanceFromEnd_max = gfc[node][constants._iNDistancesEnd_max]
    distanceFromEnd_avg = gfc[node][constants._iNDistancesEnd_avg]    
    
    arcPrimSource = gfc[node][constants._iNPrimitiveNodeSource]
    arcPrimTarget = gfc[node][constants._iNPrimitiveNodeTarget]

    sourcesNodeIsPrimitive = gfc[node][constants._iNSourcesIsPrimitive]
    targetsNodeIsPrimitive = gfc[node][constants._iNTargetIsPrimitive]

    return source, target, \
        distanceFromBegin, distanceFromBegin_min, distanceFromBegin_max, distanceFromBegin_avg, \
        distanceFromEnd, distanceFromEnd_min, distanceFromEnd_max, distanceFromEnd_avg, \
        arcPrimSource, arcPrimTarget, sourcesNodeIsPrimitive, targetsNodeIsPrimitive

def getPrimitiveNode(primitiveNodes):
    nodes = primitiveNodes[2: ]
    nodes = str(nodes).replace("arco (", "").replace(")", "")

    primitiveNodes = []

    nodes = str(nodes).splitlines()
    for node in nodes:
        if str(node).__contains__(","):
            sourceTarget = str(node).split(",")
            source = sourceTarget[0]
            target = sourceTarget[1]
            primitiveNodes.append([source, target])

    return primitiveNodes

def calcDistanceFromEnd(gfc, numNodes):
    distance = 0

    # Serve para atribuir os nós finais
    endNodes = []

    # Atribui distância 0 aos nós finais
    for endNode in gfc:
        if gfc[endNode][constants._iNTargets] == []:
            gfc[endNode][constants._iNDistancesEnd] = [distance]
            endNodes.append(endNode)
   
    repeat = True

    # Caso não seja possível calcular a distância até o início de todos nós na primeira vez, refaz o cálculo
    while repeat:
        repeat = False        

        # Percorre todos os nós do grafo de forma decrescente
        for nodeCount in range(numNodes, 0, -1):
            # Verifica se o numeral corresponde a um nó do grafo
            if str(nodeCount) not in gfc:
                continue
            node = str(nodeCount)

            # A distância dos nós finais já foi calculada, portanto ignora-os
            if node not in endNodes:
                
                nodeDistances = []

                # Percorre todos os nós destinos do nó desejado
                for iCount in range(len(gfc[node][constants._iNTargets])):
                    nodeTarget = gfc[node][constants._iNTargets][iCount]

                    # Verifica se a distância do nó destino até o final já foi calculada
                    if gfc[str(nodeTarget)][constants._iNDistancesEnd] != -1 and len(gfc[str(nodeTarget)][constants._iNDistancesEnd]) > 0:
                        #print('Calculando distância do nó {} até o fim pelo {} ({})'.format(node, nodeTarget, len(gfc[str(nodeTarget)][_iNDistancesEnd])))
                        for targetDistances in gfc[str(nodeTarget)][constants._iNDistancesEnd]:
                            #print('Calculando distância do nó {} até o fim pelo {} ({})'.format(node, nodeTarget, targetDistances))
                            nodeDistances.append(int(targetDistances) + 1)
                    else:
                        repeat = True
                        #print('Não foi possível calcular o nó {} pelo {}'.format(node, nodeTarget))

                gfc[node][constants._iNDistancesEnd] = nodeDistances

    return gfc

def calcDistanceFromBegin(gfc):
    distance = 0

    # Atribui distância 0 ao nó 1
    gfc["1"][constants._iNDistancesBegin] = [distance]

    iCount = 1

    repeat = True

    # Caso não seja possível calcular a distância até o início de todos nós na primeira vez, refaz o cálculo
    while repeat:
        repeat = False
        
        # Percorre todos os nós do grafo
        for node in gfc:
            # A distância do nó 1 já foi calculada, portanto ignora-se
            if not node == "1":
                
                nodeDistances = []

                # Percorre todos os nós origens do nó desejado
                for iCount in range(len(gfc[node][constants._iNSources])):
                    nodeSource = gfc[node][constants._iNSources][iCount]

                    # Verifica se a distância do nó origem até o início já foi calculada
                    if not gfc[str(nodeSource)][constants._iNDistancesBegin] == -1:
                        for sourceDistances in gfc[str(nodeSource)][constants._iNDistancesBegin]:
                            #print('Calculando distância do nó {} até o início pelo {} ({})'.format(node, nodeSource, sourceDistances))
                            nodeDistances.append(int(sourceDistances) + 1)
                    else:
                        repeat = True
                        #print('Não foi possível calcular o nó {} pelo {}'.format(node, nodeSource))

                gfc[node][constants._iNDistancesBegin] = nodeDistances

            iCount += 1

    return gfc

def getSource(gfc, node):
    source = []

    iCount = 1
    # Percorre todos os nós do GFC
    for _node in gfc:

        # Verifica se o nó desejado pertence ao conjunto de nós destino do nó do laço
        if node in gfc[_node][constants._iNTargets]:
            source.append(int(_node))
        
        iCount += 1
    
    return source   

def getTarget(gfc, node):
    target = []
    
    for targetNode in gfc[node][constants._iNTargets]:
        target.append(int(targetNode))

    return target

def prepareGFC(gfc, primitiveNodes):
    #GFC
        # Key = Nó
        # Value =   0 - Destinos | _iNTargets
        #           1 - Origens | _iNSources
        #           2 - Distância início | _iNDistancesBegin
        #           3 - Distância fim | _iNDistancesEnd


    gfc = str(gfc).splitlines()
    
    numNodes = int(gfc[0])

    gfc = gfc[2:]

    newGFC = {}
    last = ""
    for infoGFC in gfc:
        if not infoGFC.__contains__(' ') and not infoGFC == '0':
            last = infoGFC
        else:
            if int(last) <= numNodes:
                infonode = []
                infonode.append(list(infoGFC.split(' ')))                       #_iNTargets
                infonode.append([])                                             #_iNSources
                infonode.append(-1)                                             #_iNDistancesBegin
                infonode.append(-1)                                             #_iNDistancesBegin_min
                infonode.append(-1)                                             #_iNDistancesBegin_max
                infonode.append(-1)                                             #_iNDistancesBegin_avg
                infonode.append(-1)                                             #_iNDistancesEnd
                infonode.append(-1)                                             #_iNDistancesEnd_min
                infonode.append(-1)                                             #_iNDistancesEnd_max
                infonode.append(-1)                                             #_iNDistancesEnd_avg
                infonode.append(isPrimitiveNodeSource(primitiveNodes, last))    #_iNPrimitiveNodeSource
                infonode.append(isPrimitiveNodeTarget(primitiveNodes, last))    #_iNPrimitiveNodeTarget
                infonode.append([])                                             #_iNSourcesIsPrimitive
                infonode.append([])                                             #_iNTargetIsPrimitive
                newGFC[last] = infonode
                newGFC[last][constants._iNTargets].remove('0')

    # Calcula as origens
    for infoGFC in newGFC:
        newGFC[infoGFC][constants._iNSources] = getSource(newGFC, infoGFC)

    calcDistanceFromBegin(newGFC)
    calcDistanceFromEnd(newGFC, numNodes)

    for infoGFC in newGFC:

        # Calcula as distâncias mínimas, máximas e médias até o início
        minDistanceB, maxDistanceB, averageDistanceB = calcMinMaxAvgDistances(newGFC, infoGFC, constants._iNDistancesBegin)
        newGFC[infoGFC][constants._iNDistancesBegin_min] = minDistanceB
        newGFC[infoGFC][constants._iNDistancesBegin_max] = maxDistanceB
        newGFC[infoGFC][constants._iNDistancesBegin_avg] = averageDistanceB

        # Calcula as distâncias mínimas, máximas e médias até o fim
        minDistanceE, maxDistanceE, averageDistanceE = calcMinMaxAvgDistances(newGFC, infoGFC, constants._iNDistancesEnd)
        newGFC[infoGFC][constants._iNDistancesEnd_min] = minDistanceE
        newGFC[infoGFC][constants._iNDistancesEnd_max] = maxDistanceE
        newGFC[infoGFC][constants._iNDistancesEnd_avg] = averageDistanceE

        #Verifica se os nós origens fazem parte do arco primitivo
        for sourceNode in newGFC[infoGFC][constants._iNSources]:
            isPrimitive = isPrimitiveNodeSource(primitiveNodes, sourceNode) or isPrimitiveNodeTarget(primitiveNodes, sourceNode)
            newGFC[infoGFC][constants._iNSourcesIsPrimitive].append(1 if isPrimitive else 0)

        #Verifica se os nós destinos fazem parte do arco primitivo
        for targetNode in newGFC[infoGFC][constants._iNTargets]:
            isPrimitive = isPrimitiveNodeSource(primitiveNodes, targetNode) or isPrimitiveNodeTarget(primitiveNodes, targetNode)
            newGFC[infoGFC][constants._iNTargetIsPrimitive].append(1 if isPrimitive else 0)

    return newGFC, numNodes

def calcMinMaxAvgDistances(newGFC, infoGFC, distance):
    minDistance = None
    maxDistance = None
    average = None

    totalDistance = 0
    count = 0
    
    for distanceBegin in newGFC[infoGFC][distance]:
        count += 1
        
        intDistanceBegin = int(distanceBegin)
        totalDistance += intDistanceBegin
        if minDistance == None and maxDistance == None:
            minDistance = intDistanceBegin
            maxDistance = intDistanceBegin
        else:
            if intDistanceBegin < minDistance:
                minDistance = intDistanceBegin
            if intDistanceBegin > maxDistance:
                maxDistance = intDistanceBegin

    average = totalDistance / count

    return minDistance, maxDistance, average

def isPrimitiveNodeSource(primitiveNodes, node):
    for primitiveNode in primitiveNodes:
        if str(node) in primitiveNode[0]:
            return True
    
    return False

def isPrimitiveNodeTarget(primitiveNodes, node):
    for primitiveNode in primitiveNodes:
        if str(node) in primitiveNode[1]:
            return True
    
    return False

def gfcMain(gfcFile, arcPrimFile, showOutput = False):
    baseFolder = gfcFile[0: str(gfcFile).rindex("/")]
    unitName = gfcFile[str(gfcFile).rindex("/") + 1 : str(gfcFile).rindex(".")]

    gfc, numNodes = getGfc(gfcFile, arcPrimFile, showOutput)

    dotFileName = "{baseFolder}/{unitName}.dot".format(baseFolder = baseFolder, unitName = unitName)
    gfcToDot(gfc, dotFileName)

    pngFileName = "{baseFolder}/{unitName}.png".format(baseFolder = baseFolder, unitName = unitName)
    dotToPng(dotFileName, pngFileName)

    return gfc, numNodes

def gfcToDot(gfc, outputDotFile):
    contentDotFile = "Digraph G {\n"

    for node in gfc:
        for nodeTarget in gfc[node][constants._iNTargets]:
            contentDotFile = "{contentDotFile}\t{node} -> {nodeTarget};\n".format(
            contentDotFile = contentDotFile, node = node, nodeTarget = nodeTarget)

    contentDotFile = "{contentDotFile}}} ".format(contentDotFile = contentDotFile)
    util.write(outputDotFile, contentDotFile)

def dotToPng(dotFileName, pngFileName):
    command = "dot -Tpng -o{pngFileName} {dotFileName}".format(
        pngFileName = pngFileName, dotFileName = dotFileName)
        
    subprocess.call(command, shell=True) 

if __name__ == '__main__':
    #   GFC File Name
    #   Arc Prim File Name
    #   ShowOutput (optional)
    if len(sys.argv) > 3:
        gfcFileName = sys.argv[1]
        arcPrimFile = sys.argv[2]
        showOutput = sys.argv[3]
        gfcMain(gfcFileName, showOutput)
    elif len(sys.argv) > 2:
        gfcFileName = sys.argv[1]
        arcPrimFile = sys.argv[2]
        gfcMain(gfcFileName, arcPrimFile)
    else:
        print ('##### Exit #####')    