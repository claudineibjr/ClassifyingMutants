# Shapiro-Wilk Test
# https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from scipy.stats import shapiro
import sys 
from matplotlib import pyplot

from utils import getData, getFullData, getDataByFile

from util import getPossibleClassifiers

def runShapiroTest(data, alpha):
    stats, pValue = shapiro(data)
    print('Statistics: {} | pValue: {} | Is Parametric: {}'.format(stats, pValue, pValue > alpha))

    return pValue > alpha


def testDataDistribution(metric):
    minimalData, equivalentData = getDataByFile('{}/ML/Results/Summary/Summary_BestClassifiers_All30Runs.csv'.format(os.getcwd()), 'F1')

    # ============================================
    # ===== Test Data distribution ===============
    # ============================================
    for model, data in [['MINIMAL', minimalData], ['EQUIVALENT', equivalentData]]:
        for classifier in getPossibleClassifiers():
            print('{} - {}'.format(model, classifier))
            runShapiroTest(data.query('Classifier == \'{}\''.format(classifier))[metric], 0.05)
            print('')
        print('')

    # F1
        # Equivalent
            #  'KNN', 'DT', 'RF', 'SVM', 'LDA', 'LR' and 'GNB' are parametric

        # Minimal
            #  'KNN', 'DT', 'RF', 'SVM', 'LDA' and 'LR' are parametric
            #  'GNB' is nonparametric

if __name__ == '__main__':
    testDataDistribution('F1')