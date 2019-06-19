# -*- coding: utf-8 -*-

from __future__ import division, print_function

import gc
import math
from multiprocessing import Pool

import numpy as np


class Feature(object):
    def __init__(self, res1, res2):
        self.res1 = res1
        self.res2 = res2

    def prim_seq(self):
        blosum62 = np.array([
            [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],
            [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],
            [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],
            [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],
            [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
            [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],
            [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],
            [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],
            [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],
            [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],
            [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],
            [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],
            [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],
            [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],
            [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],
            [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],
            [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],
            [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],
            [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],
            [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4]])

        aadict = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9, 'L': 10,
                  'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19}

        idx1 = aadict[self.res1.seq]
        idx2 = aadict[self.res2.seq]
        prim_seq_score = (blosum62[idx1][idx2] + 4.) / 15.

        return prim_seq_score

    def physical(self):
        factorarr = np.array([[0., 0.07142857, 0.17534247, 0.37790698, 0.20810811, 0.28475336, 0.49560117, 0.47639485,
                               0.62593516, 0.46717172],
                              [0.49171271, 0.7955665, 0.81643836, 1., 0., 0.56278027, 0.82404692, 0.40987124,
                               0.63092269, 0.82323232],
                              [0.74585635, 0.46551724, 0.40821918, 0.69186047, 0.50810811, 0.5426009, 0.52785924,
                               0.75751073, 0.8478803, 0.15151515],
                              [0.59116022, 0.42857143, 0.00821918, 0.69186047, 0.21081081, 0.49327354, 0.1085044,
                               0.5944206, 0.76309227, 0.76515152],
                              [0.4640884, 0.2635468, 0.56438356, 0.15116279, 0.26756757, 1., 1., 0.34549356, 0.8553616,
                               0.86616162],
                              [0.30110497, 0.54187192, 0.46027397, 0.77616279, 0.75675676, 0.59192825, 0.80058651,
                               0.34120172, 0.56608479, 0.],
                              [0.03038674, 0.52955665, 0., 0.79651163, 0.10540541, 0.54932735, 0.5659824, 0.5751073,
                               0.48628429, 0.55808081],
                              [0.83425414, 0., 0.37808219, 0.40988372, 0.48648649, 0.43497758, 0.94134897, 1., 0.159601,
                               0.70454545],
                              [0.31767956, 0.61083744, 0.36438356, 0.5377907, 0.89459459, 0.68609865, 0.01173021,
                               0.5944206, 0.8553616, 1.],
                              [0.22928177, 0.44334975, 0.93150685, 0.23255814, 0.31351351, 0.46636771, 0.31085044,
                               0.60300429, 0.73815461, 0.13888889],
                              [0.14364641, 0.48275862, 0.37534247, 0.13662791, 0.31081081, 0., 0.83577713, 0.3304721,
                               0.68578554, 0.82323232],
                              [0.33701657, 0.68472906, 0.37808219, 0.9505814, 0.87567568, 0.09641256, 0.8914956,
                               0.47639485, 0.45386534, 0.73989899],
                              [0.0441989, 0.5270936, 0.3260274, 0.24418605, 1., 0.80044843, 0.63049853, 0.51716738,
                               0.25685786, 0.65656566],
                              [0.37292818, 0.72413793, 0.34246575, 0.04069767, 0.51891892, 0.27802691, 0.75073314,
                               0.72961373, 1., 0.47727273],
                              [1., 0.40147783, 0.1260274, 0.23837209, 0.6972973, 0.35874439, 0.64222874, 0., 0.75810474,
                               0.51767677],
                              [0.65469613, 0.21674877, 0.48493151, 0.57848837, 0.4027027, 0.3632287, 0., 0.24678112,
                               0.33167082, 0.53030303],
                              [0.50276243, 0.31034483, 0.77260274, 0.63953488, 0.43243243, 0.50672646, 0.62463343,
                               0.24678112, 0.43391521, 0.63636364],
                              [0.51381215, 1., 0.24383562, 0., 0.14594595, 0.58744395, 0.41348974, 0.40772532, 0.,
                               0.43686869],
                              [0.8121547, 0.84729064, 0.66027397, 0.29360465, 0.45945946, 0.30717489, 0.46334311,
                               0.71459227, 0.56109726, 0.72222222],
                              [0.22651934, 0.30788177, 1., 0.34011628, 0.59459459, 0.27802691, 0.24046921, 0.50643777,
                               0.45885287, 0.75252525]])

        aadict = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9, 'L': 10,
                  'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19}

        physical_score = []

        idx1 = aadict[self.res1.seq]
        idx2 = aadict[self.res2.seq]
        physical_score.extend(factorarr[idx1])
        physical_score.extend(factorarr[idx2])

        return physical_score

    def asa(self):
        maxasa = {'A': 129.0, 'R': 274.0, 'N': 195.0, 'D': 193.0, 'C': 167.0, 'E': 223.0, 'Q': 225.0,
                  'G': 104.0, 'H': 224.0, 'I': 197.0, 'L': 201.0, 'K': 236.0, 'M': 224.0, 'F': 240.0,
                  'P': 159.0, 'S': 155.0, 'T': 172.0, 'W': 285.0, 'Y': 263.0, 'V': 174.0}
        rasa = []

        rasa.append(self.res1.asa[0] / maxasa[self.res1.seq])
        rasa.append(self.res2.asa[0] / maxasa[self.res2.seq])

        return rasa

    def dihedral_angle(self, ang1, ang2):
        angles = []

        angle = ang1
        sinval = math.sin(math.radians(float(angle)))
        cosval = math.cos(math.radians(float(angle)))
        angles.append((sinval + 1.) / 2.)
        angles.append((cosval + 1.) / 2.)
        angle = ang2
        sinval = math.sin(math.radians(float(angle)))
        cosval = math.cos(math.radians(float(angle)))
        angles.append((sinval + 1.) / 2.)
        angles.append((cosval + 1.) / 2.)

        return angles

    def theta_angle(self):
        angles = []

        angle = self.res1.theta[0]
        cosval = math.cos(math.radians(float(angle)))
        angles.append((cosval + 1.) / 2.)
        angle = self.res2.theta[0]
        cosval = math.cos(math.radians(float(angle)))
        angles.append((cosval + 1.) / 2.)

        return angles

    def get_results(self):
        features = []

        # Primary Sequence
        features.append(self.prim_seq())
        # Physical Property
        features.extend(self.physical())
        # Secondary Structure
        features.extend(self.res1.ss)
        features.extend(self.res2.ss)
        # ASA
        features.extend(self.asa())
        # PHI
        features.extend(self.dihedral_angle(self.res1.phi[0], self.res2.phi[0]))
        # PSI
        features.extend(self.dihedral_angle(self.res1.psi[0], self.res2.psi[0]))
        # THETA
        features.extend(self.theta_angle())
        # TAU
        features.extend(self.dihedral_angle(self.res1.tau[0], self.res2.tau[0]))

        # data dim for each residue pair is 43
        return np.array(features, dtype=np.float32)


def _calc_feature_process(info):
    db_chain, target_frags = info
    db_frags = db_chain.get_frag(fraglen=1)
    target_len = len(target_frags)
    results_per_chain = []

    for i in range(db_chain.length):
        results_per_chain.append([])
        for j in range(target_len):
            results_per_chain[i].append(Feature(target_frags[j], db_frags[i]).get_results())

    return np.array(results_per_chain, dtype=np.float32)


class CalcFeature(object):
    def __init__(self, targetprot, db):
        self.TargetProt = targetprot
        self.TargetResInfo = []
        self.DB = db
        self.FeatsMatrix = []

    def get_target_res_info(self):
        self.TargetResInfo = self.TargetProt.get_frag(fraglen=1)

    def get_feats_result(self):
        self.get_target_res_info()
        pool = Pool(20)
        info = zip(self.DB, [self.TargetResInfo] * len(self.DB))
        self.FeatsMatrix = pool.map(func=_calc_feature_process, iterable=info)

        pool.close()
        pool.join()

        gc.collect()
