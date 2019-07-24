# Posso chamar gfcUtils.py a função getOffsetFromCode
import util
import sys
import subprocess

import constants
import gfcUtils

print(gfcUtils.getOffsetFromCode('/home/claudinei/Repositories/RelationshipBetweenMutationAndGFC/Programs_new/LRS/__LRS.c', 25413, 40))
print(gfcUtils.getOffsetFromCode('/home/claudinei/Repositories/RelationshipBetweenMutationAndGFC/Programs_new/LRS/__LRS.c', 25985, 32))