# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os

import rmsd

from fragment import write_file


class OutputResults(object):
    def __init__(self, targetprot, database, AllCandids, output_dir="examples/outputs/", calcrmsd=False):
        self.TargetProt = targetprot
        self.database = database
        self.fragmentnum = self.TargetProt.length-7+1
        self.AllCandids = AllCandids
        self.FinalCandids = [[] for _ in range(self.fragmentnum)]

        self.Output_Dir = output_dir

        self.flag_calcRMSD = calcrmsd

        self.Results_lib = []
        if self.flag_calcRMSD:
            head = 'Position\tSeq\tPDBName\tLocIdx\tValue\tpredRMSD\tnRMSD\n'
            self.RMSD = rmsd.calc_rmsd(self.TargetProt, self.database)
        else:
            head = 'Position\tSeq\tPDBName\tLocIdx\tValue\tpredRMSD\n'
        self.Results_lib.append(head)

        self.Results_frag = []

    def selection(self, Min_Selec_Num, Max_Selec_Num):
        # 2.0 label 90%
        cutoff = [1.0, 1.7, 1.9, 1.4, 1.6, 1.9, 2.0, 2.0, 2.0]
        # 2.0 label 85% (limit to 2.0)
        supp_cutoff = [1.4, 2.0, 2.0, 1.6, 1.9, 2.0, 2.0, 2.0, 2.0]

        allsteps_frags = [[[] for j in range(self.fragmentnum)] for i in range(3)]

        for i in range(self.fragmentnum):
            for candid in self.AllCandids[i]:
                fraglen = candid[0]
                pred_rmsd = candid[4]

                if pred_rmsd < cutoff[fraglen - 7]:
                    allsteps_frags[0][i].append(candid)
                elif pred_rmsd < supp_cutoff[fraglen - 7]:
                    allsteps_frags[1][i].append(candid)
                elif fraglen == 7:
                    allsteps_frags[2][i].append(candid)

        self.AllCandids = []

        for i in range(self.fragmentnum):
            # step1: main frag, up to max_selec_num
            allsteps_frags[0][i].sort(key=lambda x: x[4])
            num = len(allsteps_frags[0][i])
            self.FinalCandids[i].extend(allsteps_frags[0][i][:min(Max_Selec_Num, num)])

            num = len(self.FinalCandids[i])

            if num < Min_Selec_Num:
                # step2: supp frag, up to min_selec_num
                allsteps_frags[1][i].sort(key=lambda x: x[4])
                supp_num = min(Min_Selec_Num - num, len(allsteps_frags[1][i]))
                self.FinalCandids[i].extend(allsteps_frags[1][i][:supp_num])

                # step3: final selection by val, up to min_selec_num
                allsteps_frags[2][i].sort(key=lambda x: x[3], reverse=True)
                selec_num = Min_Selec_Num - num
                self.FinalCandids[i].extend(allsteps_frags[2][i][:(selec_num)])

    def output(self):
        for i in range(self.fragmentnum):
            head = ' position:          %3d neighbors:          %3d\n\n' % (i + 1, len(self.FinalCandids[i]))
            self.Results_frag.append(head)

            rmsd_res = []
            if self.flag_calcRMSD:
                rmsd_res = self.RMSD.get_results(i, self.FinalCandids[i])

            for j, candid in enumerate(self.FinalCandids[i]):
                fraglen, pdbidx, fraglocidx, val, pred_rmsd = candid

                pdbname = self.database[pdbidx].name
                fragseq = self.database[pdbidx].seq[fraglocidx:fraglocidx+fraglen]
                fragss = self.database[pdbidx].ss_str[fraglocidx:fraglocidx+fraglen]
                fragss = fragss.replace('C', 'L')
                fragtorangs = self.database[pdbidx].torangs[fraglocidx:fraglocidx+fraglen]
                fragcoords = self.database[pdbidx].coords[fraglocidx:fraglocidx+fraglen]

                # record for lib
                if self.flag_calcRMSD:
                    line = '%d\t%s\t%s\t%d\t%.3f\t%.3f\t%.3f\n' % (
                        i, fragseq, pdbname, fraglocidx, val, pred_rmsd, rmsd_res[j])
                else:
                    line = '%d\t%s\t%s\t%d\t%.3f\t%.3f\n' % (
                        i, fragseq, pdbname, fraglocidx, val, pred_rmsd)
                self.Results_lib.append(line)

                # output frag
                for k in range(fraglen):
                    phi, psi, omega = fragtorangs[k] 
                    coord_x, coord_y, coord_z = fragcoords[k]
                    line = ' ' + pdbname[:4].lower() + ' ' + pdbname[4].upper() + '  ' \
                           + '%4d %s %s %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n' % (
                               fraglocidx+k, fragseq[k], fragss[k], phi, psi, omega, coord_x, coord_y, coord_z)
                    self.Results_frag.append(line)
                self.Results_frag.append('\n')

        # write files
        if not os.path.exists(self.Output_Dir):
            os.makedirs(self.Output_Dir)

        result_lib_dir = os.path.join(self.Output_Dir, '{}.lib'.format(self.TargetProt.name))
        write_file(result_lib_dir, self.Results_lib, 'w')

        result_frag_dir = os.path.join(self.Output_Dir, '{}.frag'.format(self.TargetProt.name))
        write_file(result_frag_dir, self.Results_frag, 'w')

