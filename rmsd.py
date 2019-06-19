# -*- coding: utf-8 -*-

from __future__ import division, print_function

import copy
from multiprocessing import Pool

import numpy as np


class RMSD(object):
    """
    Calculate Root-mean-square deviation (RMSD) of Two Molecules Using Rotation. 
    Rotate matrix P unto Q using Kabsch algorithm and calculate the RMSD.
    
    The algorithm works in three steps:
        - a translation of P and Q
        - the computation of a covariance matrix C
        - computation of the optimal rotation matrix U
        http://en.wikipedia.org/wiki/Kabsch_algorithm
    
    Parameters
    ----------
    p : numpy array
        (N,D) matrix, where N is points and D is dimension.
    q : numpy array
        (N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    rmsd : float
        root-mean squared deviation
    """

    def __init__(self, p, q):
        self.P = copy.deepcopy(p)
        self.Q = copy.deepcopy(q)

        self.N, self.D = self.P.shape

        self.U = np.ones((self.D, self.D), dtype='float32')

        self.rmsd = -1.
        self.nrmsd = -1.

    def centroid(self, x):
        """
        Calculate the centroid from a vectorset X.    
        C = sum(X)/len(X)
        """
        c = x.mean(axis=0)
        return c

    def kabsch_p(self):
        """
        U : matrix
            Rotation matrix (D,D)
        """

        # Computation of the covariance matrix
        c = np.dot(np.transpose(self.P), self.Q)

        # Computation of the optimal rotation matrix
        # This can be done using singular value decomposition (SVD)
        # Getting the sign of the det(V)*(W) to decide
        # whether we need to correct our rotation matrix to ensure a
        # right-handed coordinate system.
        # And finally calculating the optimal rotation matrix U
        # see http://en.wikipedia.org/wiki/Kabsch_algorithm
        v, s, w = np.linalg.svd(c)
        d = (np.linalg.det(v) * np.linalg.det(w)) < 0.0

        if d:
            s[-1] = -s[-1]
            v[:, -1] = -v[:, -1]

        # Create Rotation matrix U
        self.U = np.dot(v, w)

    def kabsch_rotate(self):
        self.kabsch_p()

        # Rotate P
        self.P = np.dot(self.P, self.U)

    def rmsd_(self):    
        rmsd = 0.0
        for p, q in zip(self.P, self.Q):
            rmsd += sum([(p[i] - q[i])**2.0 for i in range(self.D)])
        return np.sqrt(rmsd/self.N)

    def calc_rmsd(self):
        # trasition
        p_c = self.centroid(self.P)
        q_c = self.centroid(self.Q)
        self.P -= p_c
        self.Q -= q_c

        self.kabsch_rotate()  # rotate P unto Q
        return self.rmsd_()

    def calc_nrmsd(self):
        if self.rmsd == -1:
            self.rmsd = self.calc_rmsd()
        self.nrmsd = self.rmsd/(0.270724+0.0729276*self.N)

        return self.nrmsd


def _calc_rmsd_process(coords_pair):
    p, q = coords_pair
    rmsd_obj = RMSD(p, q)
    nrmsd = rmsd_obj.calc_nrmsd()
    return nrmsd
    

class calc_rmsd(object):
    def __init__(self, targetprot, database):
        self.targetprot = targetprot
        self.database = database

    def get_results(self, target_locidx, candids_info):
        target_frag_coords = []
        candids_frag_coords = []

        for candid in candids_info:
            fraglen = candid[0]
            pdbidx = candid[1]
            locidx = candid[2]

            coords = self.targetprot.coords[target_locidx:target_locidx+fraglen]
            coords = coords.reshape(-1, 3)
            coords = coords.astype(np.float32)
            target_frag_coords.append(coords)

            coords = self.database[pdbidx].coords[locidx:locidx+fraglen]
            coords = coords.reshape(-1, 3)
            coords = coords.astype(np.float32)
            candids_frag_coords.append(coords)

        pool = Pool(20)
        results = pool.map(func=_calc_rmsd_process, iterable=zip(target_frag_coords, candids_frag_coords))
        pool.close()
        pool.join()

        return results
