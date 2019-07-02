#!/usr/bin/python
import os
import sys
import math
import pdb
#from progs import listProgs


# Esse programa calcula TODOS os mutantes que nao sao incluidos por
# outros mutantes. Ou seja, aquele que sao minimais e todos os seus
# indistinguiveis
#Alterado em 4 de abril de 2018

def computeMinimal(prog, generateReport = True, baseFolder = None):
    #os.chdir(prog)
    if (generateReport):
        statement = "report -trace -L 2 " + prog
        os.system(statement)
    report = prog+".trc"
    repfile = open(report, "r")
    cont = -2
    mutants = [] # list of mutants still to analyze
    hashset = dict()
    for line in repfile:
        if (cont < 0 or line.find("TOTAL") >= 0):
            cont += 1
            continue
        tcLine = line.split()
        #print line
        #print tcLine
        s =  set()
        i = 0
        for tc in tcLine:
            k = int(tc)
            if ( k > 1 and k < 6 ): # mutant is dead
                s.add(i)  # tc is includeed in teh mutant set
            i += 1
        if (len(s) > 0): # equiv mutants have size 0
            hashset[cont] = s
            mutants.append(cont)
        cont += 1
    # at this point each mutant has an entry in the list mutants
    # and in the hashset variable. hashset holds the test cases
    # that kill each mutant
    
    before = len(mutants)
    counter = dict()
    
    print(prog + " had " + str(before) + " mutants ", end=' ')
    for m in mutants[:]:
        if ( m not in hashset):
            continue
        s = hashset[m]
        counter[m] = 0
        remove = set()
        for m2 in mutants[:]:
            if (m2 == m):
                continue
            if ( m2 not in hashset):
                continue
            s2 = hashset[m2]
            if (s < s2): #m subsumes m2 
                #print "Mutant " + str(m) + " subsumes "  + str(m2)
                #print s
                #print s2
                remove.add(m2)
                counter[m] = counter[m] + 1
        for m2 in remove:
            mutants.remove(m2)
            hashset.pop(m2)
    after = len(mutants)
    print("ended with " + str(after) +  " (" + "%5.2f" % (float(after)/ float(before) * 100.0) + "%)")
    
    if (baseFolder == None):
        baseFolder = prog[0: str(prog).rfind("/")]

    minFile = open("{}/minimal.txt".format(baseFolder), "w")
    for m in mutants:
        minFile.write(str(m) + '\n')
    minFile.close()

    soma = 0.0
    max = 0
    min = 1000000
    # this is number of test cases that kill each mutant
    minFile = open("{}/minimal-sizes.txt".format(baseFolder), "w")
    for m in mutants:
        s = len(hashset[m])
        soma += s
        if (s < min):
            min = s
        if (s > max):
            max = s
        minFile.write(str(m) + " " + str(s) + "\n")
    minFile.write("Min: "+ str(min)+ "\n") 
    minFile.write("Max: "+ str(max)+ "\n")
    minFile.write("Avg: "+ str(soma/float(len(mutants))) + "\n")
    minFile.close()

    soma = 0.0
    max = 0
    min = 1000000
    # this is number of mutants subsumed by each minimal mutant
    minFile = open("{}/minimal-subsume-sizes.txt".format(baseFolder), "w")
    for m in mutants:
        s = counter[m]
        soma += s
        if (s < min):
            min = s
        if (s > max):
            max = s
        minFile.write(str(m) + " " + str(s) + "\n")
    minFile.write("Min: "+ str(min)+ "\n") 
    minFile.write("Max: "+ str(max)+ "\n")
    minFile.write("Avg: "+ str(soma/float(len(mutants))) + "\n")
    minFile.close()

    
    os.chdir("..")
    return



if __name__=="__main__":                       # If this script is run as a program:
    
    if ( len(sys.argv) > 1 ):
        prog = sys.argv[1]
        computeMinimal(prog)
    #else:
    #    programs = listProgs()
    #    for prog in programs:
    #        computeMinimal(prog)