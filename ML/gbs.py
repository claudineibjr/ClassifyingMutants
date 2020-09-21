import sys
import os

sys.path.insert(0, '{}/ML'.format(os.getcwd()))
from ML_Mutants import debug_main

import util
from util import getPossibleClassifiers, getPossibleTargetColumns


def getDropableColumns():
    possibleDropableColumns = [
        '_IM_OPERATOR', '_IM_SOURCE_PRIMITIVE_ARC', '_IM_TARGET_PRIMITIVE_ARC',
        '_IM_DISTANCE_BEGIN_MIN', '_IM_DISTANCE_BEGIN_MAX',
        '_IM_DISTANCE_BEGIN_AVG', '_IM_DISTANCE_END_MIN',
        '_IM_DISTANCE_END_MAX', '_IM_DISTANCE_END_AVG', '_IM_COMPLEXITY',
        '_IM_TYPE_STATEMENT'
    ]
    return possibleDropableColumns


def getMaxF1(fileName):
    content = util.splitFileInColumns(fileName, ';')
    return content[1][4]


def experiment():
    baseFolder = '{}/ML/Results'.format(os.getcwd())

    targetColumns = getPossibleTargetColumns()
    targetColumns.sort()

    classifiers = getPossibleClassifiers()
    classifiers.sort()

    dropableColumns = getDropableColumns()
    dropableColumns.sort()

    resultsFile = []

    for targetColumn in targetColumns:
        for classifier in classifiers:
            originalFile = '{}/{}/{}.csv'.format(baseFolder, targetColumn,
                                                 classifier)
            columnsResult = []
            f1 = float(getMaxF1(originalFile))
            columnsResult.append(f1)
            print('{} - {} - Original: {:.2f}'.format(targetColumn, classifier,
                                                      f1))

            for column in dropableColumns:
                columnFile = '{}/{}/{} - gbs_[\'{}\'].csv'.format(
                    baseFolder, targetColumn, classifier, column)
                f1 = float(getMaxF1(columnFile))
                columnsResult.append(f1)
                print('{} - {} - {}: {:.2f}'.format(targetColumn, classifier,
                                                    column, f1))

            resultsFile.append([targetColumn, classifier, columnsResult])

    util.writeInCsvFile('{}/ML/Results/gbs.csv'.format(os.getcwd()),
                        resultsFile)


if __name__ == '__main__':
    experiment()