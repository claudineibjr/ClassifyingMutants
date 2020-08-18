import sys
import os

sys.path.insert(0, '{}/ML'.format(os.getcwd()))
from ML_Mutants import debug_main

import util
from util import getPossibleClassifiers, getPossibleTargetColumns

def experiment():
    arguments = ['']

    for classifier in getPossibleClassifiers():
        for targetColumn in getPossibleTargetColumns():
            print('\n\n', classifier, targetColumn)
            debug_main(['', '--column', targetColumn, '--classifier', classifier])

if __name__ == '__main__':
	experiment()