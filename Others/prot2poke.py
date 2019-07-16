#!/usr/bin/python
# -*- coding: utf-8 -*-

####################################################################################################
#										   prot2poke                                               #
####################################################################################################
# Descricao: Este script transforma um arquivo GFC gerado pela ProteumIM em um arquivo GFC no      #
#	 		 do gerador de arcos primitivos                                                        #
#																			   					   #
# prot2poke <file-name>							      			   			                       #
# - prog-name: nome do arquivo GFC 							   				   	                   #
#	Exemplo: prot2poke cal.gfc											   			               #
#																				      			   #
# Ultima modificação: 23/11/2018 												       			   #
####################################################################################################

import sys
import os
from datetime import datetime

'''
	Este metodo ira remover todos os 'Def' e 'Use' do arquivo gfc gerado pela ProteumIM
'''
def remove_def_use(nome_arquivo, baseFolder = None):

	if 'gfc' not in nome_arquivo:
		print('ERROR: file must be in a .gfc format')
		sys.exit(1)

	if (baseFolder == None):
		baseFolder = nome_arquivo[0: str(nome_arquivo).rfind("/")]

	novo_arquivo = open("{baseFolder}/test.txt".format(baseFolder = baseFolder), 'w')

	with open(nome_arquivo, "r+") as f:
		new_f = f.readlines()
		f.seek(0)
		for line in new_f:
			if "Def:" not in line:
				if "Use:" not in line:
					print(line)
					novo_arquivo.write(line)


	f.close()
	novo_arquivo.close()

'''
	Esta funcao ira criar os arquivos gfcs para cada uma das funcaos dos programas (inclusive a main).
'''
def cria_arquivos(nome_arquivo, baseFolder = None):
	contador = 0
	nfile = object

	if (baseFolder == None):
		baseFolder = nome_arquivo[0: str(nome_arquivo).rfind("/")]

	with open("{baseFolder}/test.txt".format(baseFolder = baseFolder), "r+") as f:
		new_f = f.readlines()
		f.seek(0)

		#temp_file = open("temp.txt", "w")

		# Contar o numero de funcoes do programa
		for line in new_f:
			if "@" in line:
				contador = contador + 1

		print(contador)

		# Gerar um arquivo gfc para cada uma das funcoes
		while(contador != 0):
			for line in new_f:
				if "@" in line:
					x = line.split("@")
					x = x[1].split("\n")
					x = x[0] + ".gfc"

					nfile = open("{baseFolder}/{x}".format(baseFolder = baseFolder, x = x), "w")
					contador = contador - 1
					print(("***** Arquivo " + x + " gerado com sucesso *****"))	    				

				if "@" not in line:
					nfile.write(line)

			nfile.close()
	

	f.close()

'''
	Esta funcao ira apagar o arquivo test.txt utilizado para execucao do script
'''			
def apaga_arquivos(nome_arquivo, baseFolder = None):
	if (baseFolder == None):
		baseFolder = nome_arquivo[0: str(nome_arquivo).rfind("/")]    	

	apagar = "rm {baseFolder}/test.txt".format(baseFolder = baseFolder)
	os.system(apagar)

def prot2PokeMain(nome_arquivo):
	remove_def_use(nome_arquivo)
	cria_arquivos(nome_arquivo)
	apaga_arquivos(nome_arquivo)    	

if __name__ == '__main__':
	now = datetime.now()
	formatted_now = now.strftime('%d/%m/%Y %H:%M:%S')
	
	print('#################################################')
	print('#\t   Executando script prot2poke\t\t#')
	print('#\t      ' + formatted_now + '\t\t#')
	print('#################################################')

	try:
		nome_arquivo = sys.argv[1]
	except:
		print('\nERROR: one arguments needed ...\nprot2poke <file-name.gfc>\n')
		sys.exit(1)  	
	
	prot2PokeMain(nome_arquivo)