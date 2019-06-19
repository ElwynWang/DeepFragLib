# -*- coding: utf-8 -*-

from __future__ import division, print_function
import numpy as np
import os
import sys

target = sys.argv[1]

with open("../../data/example/{}.spotcon".format(target), 'r') as fin:
    lines=fin.readlines()
    lens=len(lines[5].strip())
    matrix = np.zeros((lens, lens), dtype=np.float32)
    for line in lines:            
        if line[0]>='A' and line[0]<='Z':
            continue
        idx1 = int(line.split('\t')[0])
        idx2 = int(line.split('\t')[1])
        val = float(line.split('\t')[2])
        matrix[idx1][idx2] = val
        matrix[idx2][idx1] = val
    
    np.savetxt("../../data/example/{}.mat".format(target), matrix)
print ("Done! {}".format(target))   
