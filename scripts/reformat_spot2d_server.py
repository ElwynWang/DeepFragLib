# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import sys

import numpy as np

target = sys.argv[1]

with open("../../data/example/{}.spotcon".format(target), 'r') as fin:
    lines=fin.readlines()
    lens=0
    for i in range(6,21):
        if lines[i][0]>='A' and lines[i][0]<='Z':
            lens=lens+len(lines[i].strip())
        else:    
            break        

    matrix = np.zeros((lens, lens), dtype=np.float32)

    for line in lines:
        if line[0]>='A' and line[0]<='Z':
            continue
        idx1 = int(line.split('\t')[0])
        idx2 = int(line.split('\t')[1])
        val = float(line.split('\t')[4])
        matrix[idx1-1][idx2-1] = val
        matrix[idx2-1][idx1-1] = val
    
    np.savetxt("../../data/example/{}.mat".format(target), matrix)
print ("Done! {}".format(target))
