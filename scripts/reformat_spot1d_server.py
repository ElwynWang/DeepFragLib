# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import sys

target = sys.argv[1]

with open('../../data/example/{}.spot1d'.format(target), 'r') as fin:
    lines = fin.readlines()[1:]
    with open('../../data/example/{}.spd3'.format(target), 'w') as fout:
        fout.write('# SEQ SS ASA Phi Psi Theta(i-1=>i+1) Tau(i-2=>i+2) HSE_alpha_up HSE_alpha_down P(C) P(H) P(E)\n')
        for line in lines:
            line = line.split('\t')
            idx, seq, ss, asa = int(line[0]), line[1].strip(), line[2].strip(), float(line[4])
            phi, psi, theta, tau = float(line[10]), float(line[11]), float(line[8]), float(line[9])
            up, down = float(line[5]), float(line[6])
            c, e, h = float(line[12])/100., float(line[13])/100., float(line[14])/100.
            fout.write('%3d %s %s %5.1f %6.1f %6.1f %6.1f %6.1f %4.1f %4.1f %5.3f %5.3f %5.3f\n' % (idx, seq, ss, asa, phi, psi, theta, tau, up, down, c, h, e))

print ("Done! {}".format(target))    
