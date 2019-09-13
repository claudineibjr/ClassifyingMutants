import sys
import os

sys.path.insert(0, '{}/ML'.format(os.getcwd()))
from ML_Mutants import debug_main

import util

if __name__ == '__main__':
	baseResultsFolder = '{}/ML/Results/\'COLUMN\''.format(os.getcwd())
	util.renameFolder(baseResultsFolder.replace('\'COLUMN\'', 'MINIMAL'), baseResultsFolder.replace('\'COLUMN\'', 'MINIMAL_old'))
	util.renameFolder(baseResultsFolder.replace('\'COLUMN\'', 'EQUIVALENT'), baseResultsFolder.replace('\'COLUMN\'', 'EQUIVALENT_old'))
	
	arguments = ['', '--all']
	for iCount in range(30):
		# Create base results folder
		util.createFolder(baseResultsFolder.replace('\'COLUMN\'', 'MINIMAL'))
		util.createFolder(baseResultsFolder.replace('\'COLUMN\'', 'EQUIVALENT'))
		
		# Experiment
		debug_main(arguments)

		# Rename results folder to include iteration number
		util.renameFolder(baseResultsFolder.replace('\'COLUMN\'', 'MINIMAL'), baseResultsFolder.replace('\'COLUMN\'', 'MINIMAL_{}'.format(iCount + 1)))
		util.renameFolder(baseResultsFolder.replace('\'COLUMN\'', 'EQUIVALENT'), baseResultsFolder.replace('\'COLUMN\'', 'EQUIVALENT_{}'.format(iCount + 1)))

	# Rename old folder to base folder
	util.renameFolder(baseResultsFolder.replace('\'COLUMN\'', 'MINIMAL_old'), baseResultsFolder.replace('\'COLUMN\'', 'MINIMAL'))
	util.renameFolder(baseResultsFolder.replace('\'COLUMN\'', 'EQUIVALENT_old'), baseResultsFolder.replace('\'COLUMN\'', 'EQUIVALENT'))