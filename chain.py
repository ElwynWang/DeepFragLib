# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import sys
import subprocess

from fragment import *


class TargetChain(Fragment):
    def __init__(self, name=''):
        Fragment.__init__(self, name)
        self.homol = set()

    def get_target_info(self):
        filedir = os.path.join('examples', 'inputs')
        # use spider3 to get preidcted local features
        fname = os.path.join(filedir, '.'.join([self.name, 'spd3']))
        lines = read_file(fname)
        asa, phi, psi, theta, tau = [], [], [], [], []

        for i, line in enumerate(lines[1:]):
            # if len>1000, positions of all values shift right 1
            extra = 1 if i > 1000 else 0

            self.seq += line[4 + extra]
            asa.append(float(line[8 + extra:13 + extra]))

            phi.append(float(line[14 + extra:20 + extra]))
            psi.append(float(line[21 + extra:27 + extra]))
            theta.append(float(line[29 + extra:34 + extra]))
            tau.append(float(line[35 + extra:41 + extra]))

        self.length = len(self.seq)
        self.asa = np.array(asa, dtype=np.float32)
        self.phi = np.array(phi, dtype=np.float32)
        self.psi = np.array(psi, dtype=np.float32)
        self.theta = np.array(theta, dtype=np.float32)
        self.tau = np.array(tau, dtype=np.float32)

        # use psipred to get predicted ss
        fname = os.path.join(filedir, '.'.join([self.name, 'ss']))
        lines = read_file(fname)
        ss_str = ''
        ss = []
        for i, line in enumerate(lines):
            if line[5] != self.seq[i]:
                print ('Seq from psipred and seq from spider3 are not the same! Exit!')
                sys.exit(1)

            ss_str += line[7]
            h_prob = float(line[18:23])
            e_prob = float(line[25:30])
            c_prob = float(line[11:16])
            ss.append(h_prob)
            ss.append(e_prob)
            ss.append(c_prob)

        self.ss_str = ss_str
        self.ss = np.array(ss, dtype=np.float32)

    def rm_homol(self, dblabel, blast_iters, evalue_set):
        homol = []
        blastdb = ''.join(['utils/psiblast/db', dblabel, '/lib', dblabel])
        filedir = os.path.join('examples', 'inputs', '.'.join([self.name, 'fasta']))

        if not os.path.exists(filedir):
            print ("Could not find fasta file! Skip remove homologous proteins!")
            return

        comm = ' '.join(['utils/psiblast/psiblast', '-query', filedir,
                         '-db', blastdb, '-num_iterations', str(blast_iters), '-evalue', str(evalue_set)])
        ret = subprocess.Popen(comm, shell=True, stdout=subprocess.PIPE).stdout.readlines()

        if ret[36].strip().decode() != '***** No hits found *****':
            cnt = 0
            while ret[37 + cnt].strip().decode() != '':
                homol.append(ret[37 + cnt].split()[0])
                cnt += 1

        self.homol = set(homol)

    # this function only for rmsd calcluation with ca pdb file provided
    def get_target_coords(self):
        fname = os.path.join('examples', 'inputs', '.'.join([self.name, 'pdb']))
        # the last line in PDB is "END"
        lines = read_file(fname)
        if lines[-1].strip() == 'END':
            lines = lines[:-1]

        assert len(lines) == self.length
        coords = []

        for i in range(self.length):
            coords.append(float(lines[i][30:38]))
            coords.append(float(lines[i][38:46]))
            coords.append(float(lines[i][46:54]))

        self.coords = np.array(coords, dtype=np.float32).reshape(-1, 3)

    def read_target(self, calcrmsd=False):
        self.get_target_info()
        filepath = os.path.join('examples', 'inputs', '.'.join([self.name, 'pdb']))
        if calcrmsd and os.path.exists(filepath):
            self.get_target_coords()
            return True
        elif calcrmsd:
            print ("Could not find target protein pdb file! Could not calculate RMSD")
            return False

    def get_frag(self, fraglen):
        allfrags = [0] * (self.length - fraglen + 1)

        for i in range(self.length - fraglen + 1):
            allfrags[i] = cutfrag(self, i, fraglen)

        return allfrags


class DBFrag(object):
    def __init__(self):
        self.chains = []

    def build_db(self, db_label, is_rm_homol, homol):
        filedir = os.path.join(os.getcwd(), 'data')

        db_coords = read_file(os.path.join(filedir, 'DBCoords'))

        # chains in databse
        for i in range(int(db_label)):
            if is_rm_homol:
                if db_coords[15 + i * 9].strip() in homol:
                    continue

            db_chain = TargetChain()

            db_chain.name = db_coords[15 + i * 9].strip()
            db_chain.seq = db_coords[16 + i * 9].strip()
            db_chain.length = len(db_chain.seq)

            ss = db_coords[17 + i * 9].strip()
            ss = ss.replace('G', 'H')
            ss = ss.replace('I', 'H')
            ss = ss.replace('B', 'E')
            ss = ss.replace('S', 'C')
            ss = ss.replace('T', 'C')
            db_chain.ss_str = ss
            ss_prob = []
            for s in ss:
                if s == 'H':
                    ss_prob.extend([1, 0, 0])
                elif s == 'E':
                    ss_prob.extend([0, 1, 0])
                else:
                    ss_prob.extend([0, 0, 1])
            db_chain.ss = np.array(ss_prob, dtype=np.float32)

            db_chain.asa = np.array([float(x) for x in db_coords[18 + i * 9].strip().split('\t')], dtype=np.float32)

            coords = [float(x) for x in db_coords[19 + i * 9].strip().split('\t')]
            torangs = [float(x) for x in db_coords[20 + i * 9].strip().split('\t')]

            # although the torangs and the phi/psi angles of candidates are the same and duplicated,
            # we still both record them.
            db_chain.coords = np.array(coords).reshape(-1, 3)
            db_chain.torangs = np.array(torangs).reshape(-1, 3)

            db_chain.phi = np.array(torangs[0::3], dtype=np.float32)
            db_chain.psi = np.array(torangs[1::3], dtype=np.float32)

            db_chain.theta = np.array([float(x) for x in db_coords[21 + i * 9].strip().split('\t')], dtype=np.float32)
            db_chain.tau = np.array([float(x) for x in db_coords[22 + i * 9].strip().split('\t')], dtype=np.float32)

            db_chain.contact_vec = np.load("data/contact/{}.mat.npy".format(db_chain.name.lower()), allow_pickle=False)

            db_chain.candididx = len(self.chains)

            self.chains.append(db_chain)

    def get__db_frag(self, fraglen):
        allfrags = []

        for db_chain in self.chains:
            frags = db_chain.get_frag(fraglen=fraglen)
            allfrags.extend(frags)

        return allfrags
