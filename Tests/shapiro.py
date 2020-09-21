# Shapiro-Wilk Test
# https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/

from scipy.stats import shapiro
import sys 
from matplotlib import pyplot

from utils import getData, getFullData

def shapiroTests(metric):
    alpha = 0.05

    minimalData, equivalentData = getData(metric)
    # minimalData, equivalentData = getFullData(metric)

    print('\n\n=== {} ========='.format(metric))

    # === Minimal Mutants =========
    print('--- Minimal Mutants ---------')
    
    statMinimal, pMinimal = shapiro(minimalData)
    print('Statistics = %.3f, p=%.3f' % (statMinimal, pMinimal))

    if pMinimal > alpha:
        print('Sample looks Gaussian (fail to reject H0) - Is parametric')
    else:
        print('Sample does not look Gaussian (reject H0) - Is nonparametric')

    # === Minimal Mutants =========
    print('\n--- Equivalent Mutants ---------')
    
    statEquivalent, pEquivalent = shapiro(equivalentData, )
    print('Statistics = %.3f, p=%.3f' % (statEquivalent, pEquivalent))

    if pEquivalent > alpha:
        print('Sample looks Gaussian (fail to reject H0) - Is parametric')
    else:
        print('Sample does not look Gaussian (reject H0) - Is nonparametric')

def showShapiroTests():
    shapiroTests('Accuracy')
    shapiroTests('Precision')
    shapiroTests('Recall')
    shapiroTests('F1')

def getHistogramPlot(data):
    pyplot.hist(data, range=[0, 100])
    pyplot.show()

def showHistogram():
    # See histogram Plot
    minimalData, equivalentData = getData('F1')
    # minimalData, equivalentData = getFullData('F1')
    getHistogramPlot(minimalData)
    getHistogramPlot(equivalentData)

if __name__ == '__main__':
    showShapiroTests()
    # showHistogram()