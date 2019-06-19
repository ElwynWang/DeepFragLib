# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np


class Fragment(object):
    def __init__(self, name=''):
        self.name = name
        self.seq = ''
        self.length = 0.

        # for candidates, all the items are extracted from pdb file,
        # while for target, these feature are extracted from spd3 file
        self.ss_str = ''
        self.ss = np.array([], dtype=np.float32)
        self.asa = np.array([], dtype=np.float32)
        self.phi = np.array([], dtype=np.float32)
        self.psi = np.array([], dtype=np.float32)
        self.theta = np.array([], dtype=np.float32)
        self.tau = np.array([], dtype=np.float32)

        # for candidates and target (if ca pdb file provided for rmsd calculation),
        # these two features are both extracted from ca pdb file
        self.coords = np.array([], dtype=np.float32)
        self.torangs = np.array([], dtype=np.float32)

        # for candidates, this feature is extracted from true contact file,
        # while for target, this feature is the predicted information.
        self.contact_vec = np.array([], dtype=np.float32)

        self.candididx = -1


def cutfrag(chain, loc, fraglen):
    frag = Fragment()
    cutlen = fraglen

    # we only cut some key features for feature calculation
    frag.seq = chain.seq[loc:loc+cutlen]
    frag.ss = chain.ss[loc*3:(loc+cutlen)*3]
    frag.asa = chain.asa[loc:loc+cutlen]
    frag.phi = chain.phi[loc:loc+cutlen]
    frag.psi = chain.psi[loc:loc+cutlen]
    frag.theta = chain.theta[loc:loc+cutlen]
    frag.tau = chain.tau[loc:loc+cutlen]

    return frag


def calc_frag_ss(ss_str):
    fraglen = len(ss_str)

    if ss_str.count('H') >= fraglen/2.:
        ss_label = 'H'
    elif ss_str.count('E') >= fraglen/2.:
        ss_label = 'E'
    elif ss_str.count('C') >= fraglen/2.:
        ss_label = 'C'
    else:
        ss_label = 'O'

    return ss_label


def read_file(fname):
    with open(fname, 'r') as fin:
        data = fin.readlines()
        return data


def write_file(fname, contents, state):
    with open(fname, state) as fout:
        for content in contents:
            fout.write('{}'.format(content))
