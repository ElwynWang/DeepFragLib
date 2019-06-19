# -*- coding: utf-8 -*-

from __future__ import division, print_function

import gc
import os
import sys

import numpy as np

import chain
import contact
import features
import models
import outputs


if __name__ == "__main__":
    # **************** Parameters *****************#
    db_label = '956'
    evalue_set = 0.01
    blast_iters = 1
    isRmHomol = True
    isCalcRMSD = True
    Output_Dir = "examples/outputs/"
    Min_Selec_Num = 50
    Max_Selec_Num = 200
    # if you don't use GPU, please set gpu_id = ''
    gpu_id = '0'
    # *********************************************#

    TargetName = sys.argv[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    filepath = os.path.join('examples', 'inputs', '.'.join([TargetName, 'spd3']))
    if not os.path.exists(filepath):
        print ("Could not find Target Protein file! Exit!")
        sys.exit(1)

    TargetProt = chain.TargetChain(TargetName)
    _exist_pdbfie = TargetProt.read_target(calcrmsd=isCalcRMSD)
    # If there is no pdb file for rmsd calculation, then we'll skip this step.
    if not _exist_pdbfie:
        isCalcRMSD = False

    if isRmHomol:
        TargetProt.rm_homol(db_label, blast_iters, evalue_set)

    print ("Build Fragment Library for Target {}".format(TargetName))

    Database = chain.DBFrag()
    Database.build_db(db_label, isRmHomol, TargetProt.homol)

    AllFeats = features.CalcFeature(TargetProt, Database.chains)
    AllFeats.get_feats_result()

    print ("Done! Build DataBase")

    # Attention: It will need some time to load the model! Be patient!
    RegModel = models.Model_Regression()
    print ("Regression Model Loaded.")

    Contact = contact.Contact(TargetProt)
    TargetProt.contact_vec = Contact.pred_contact()
    print ("Contact Prediction Models Done.")

    AllCandids = [[] for _ in range(TargetProt.length-7+1)]

    # loop from length=7 to length=15
    for FragLen in range(7, 16):
        print ("Fragment Length = {}".format(FragLen))
        gc.collect()

        ClassifyModel = models.Model_Classify(FragLen, TargetProt, Database.chains, AllFeats, Contact, gpu_id)

        for i in range(TargetProt.length-FragLen+1):
            top_candids_info = ClassifyModel.predict(i)
            RegModel.predict(top_candids_info)

            AllCandids[i].extend(top_candids_info)

        del ClassifyModel

        print ("Done! Fragment Length = {}".format(FragLen))

    # Select final fragments and output results
    Output = outputs.OutputResults(targetprot=TargetProt, database=Database.chains, AllCandids=AllCandids,
                                   output_dir=Output_Dir, calcrmsd=isCalcRMSD)
    # select final candidates with 3 steps
    Output.selection(Min_Selec_Num, Max_Selec_Num)
    # write files
    Output.output()

    print ("Done! Build Fragment Library for Target {}".format(TargetName))
