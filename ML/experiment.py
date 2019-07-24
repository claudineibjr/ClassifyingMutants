import DecisionTree_Mutants as DecisionTree
import kNN_Mutants as kNN

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

# Utilities
import util

if __name__ == '__main__':
    print ('### BEGIN ###')
    print ('##########\t   Executing Árvore de Decisão\t ' + util.formatNow() + '\t   ##########')
    DecisionTree.main()
    
    print ('##########\t   Executing kNN\t ' + util.formatNow() + '\t   ##########')
    kNN.main()