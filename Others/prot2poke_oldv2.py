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
def remove_def_use():

	try:
		nome_arquivo = sys.argv[1]
	except:
		print('\nERROR: one arguments needed ...\nprot2poke <file-name.gfc>\n')
		sys.exit(1)

	if 'gfc' not in sys.argv[1]:
		print('ERROR: file must be in a .gfc format')
		sys.exit(1)

 	novo_arquivo = open("test.txt", 'w')

	with open(nome_arquivo, "r+") as f:
		new_f = f.readlines()
		f.seek(0)
		for line in new_f:
			if "Def:" not in line:
				if "Use:" not in line:
					print line
					novo_arquivo.write(line)


	f.close()
	novo_arquivo.close()

'''
	Esta funcao ira criar os arquivos gfcs para cada uma das funcaos dos programas (inclusive a main).
'''
def cria_arquivos():
	contador = 0
	nfile = " "

	with open("test.txt", "r+") as f:
		new_f = f.readlines()
		f.seek(0)

		#temp_file = open("temp.txt", "w")

		# Contar o numero de funcoes do programa
		for line in new_f:
			if "@" in line:
				contador = contador + 1

		print contador

		# Gerar um arquivo gfc para cada uma das funcoes
		while(contador != 0):
			for line in new_f:
				if "@" not in line:
					nfile.write(line)

				if "@" in line:
					x = line.split("@")
					x = x[1].split("\n")
					x = x[0] + ".gfc"

					nfile = open(x, "w")
					contador = contador - 1
					print("***** Arquivo " + x + " gerado com sucesso *****")	

			nfile.close()
	

	f.close()

'''
	Esta funcao ira apagar o arquivo test.txt utilizado para execucao do script
'''			
def apaga_arquivos():
	apagar = "rm test.txt"
	os.system(apagar)

if __name__ == '__main__':
	now = datetime.now()
	formatted_now = now.strftime('%d/%m/%Y %H:%M:%S')
	
	print '#################################################'
	print '#\t   Executando script prot2poke\t\t#'
	print '#\t      ' + formatted_now + '\t\t#'
	print '#################################################'
	remove_def_use()
	cria_arquivos()
	apaga_arquivos()
	
	
