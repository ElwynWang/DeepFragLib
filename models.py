# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os

import numpy as np
import tensorflow as tf

from contact import calc_contact_vec
from fragment import calc_frag_ss


class Model_Classify(object):
    def __init__(self, fraglen, targetprot, db, allfeats, contact, gpu_id):
        self.TargetProt = targetprot
        self.DB = db
        self.FragLen = fraglen
        self.AllFeats = allfeats
        self.Contact = contact
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        tf.reset_default_graph()

        model_path = "models/classify/model_len{}.pb".format(self.FragLen)
        
        with open(model_path, "rb") as f:
            output_graph_def = tf.GraphDef()
            output_graph_def.ParseFromString(f.read())

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        self.sess = tf.Session(config=config)

        self.sess.graph.as_default()
        tf.import_graph_def(output_graph_def, name='')

        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.X = self.sess.graph.get_tensor_by_name("tower_0/cpu_variables/X_Input:0")
        self.Y_ = self.sess.graph.get_tensor_by_name('tower_0/cpu_variables/layer4_sigmoid/output:0')
        self.isTraining = self.sess.graph.get_tensor_by_name("tower_0/cpu_variables/isTraining:0") 
        self.BLSTM = self.sess.graph.get_tensor_by_name("tower_0/cpu_variables/layer2_blstm/all:0")

    def predict(self, target_locidx):
        '''
        args:
            target_locidx: the location index of target protein

        return:
            top_candids_info: [pdbidx, locidx, reg_info], where reg_info is [fraglen, contact_simi, val, blstmout]
                            for regression model
            supp_torang_info: [pdbidx, locidx, torang_diff, contact_match_num, contact_match_ratio, contact_not_match_num]
            supp_contact_info: [pdbidx, locidx, torang_diff, contact_match_num, contact_match_ratio, contact_not_match_num]
        '''

        # record features and info of fragments with the same fragment ss class with target fragment
        candids_feats = []
        candids_info = []

        target_sslabel = calc_frag_ss(self.TargetProt.ss_str[target_locidx:target_locidx+self.FragLen])

        for idx1, chain in enumerate(self.DB):
            for idx2 in range(chain.length-self.FragLen+1):
                candid_ss = self.DB[idx1].ss_str[idx2:idx2+self.FragLen]
                candid_sslabel = calc_frag_ss(candid_ss)

                if candid_sslabel != target_sslabel:
                    continue

                feat = self.AllFeats.FeatsMatrix[idx1][[x for x in range(idx2, idx2+self.FragLen)],
                                                       [x for x in range(target_locidx, target_locidx+self.FragLen)]]
                candids_feats.append(feat)
                # record db_chain idx, candid_locidx
                candids_info.append([self.FragLen, idx1, idx2])

        candids_feats = np.array(candids_feats, dtype=np.float32).reshape(-1, self.FragLen, 43)

        allresult = None

        # every 10,000 samples per batch
        round_num = int(len(candids_feats)/10000)+1

        for i in range(round_num):
            if i == round_num-1:
                result = self.sess.run([self.Y_], feed_dict={self.X: candids_feats[i*10000:], self.isTraining: False})
            else:
                result = self.sess.run([self.Y_], feed_dict={self.X: candids_feats[i*10000:i*10000+10000], self.isTraining: False})
            result = result[0]
            result = 1./(1.+np.exp(-result.flatten()))
            
            if allresult is None:
                allresult = result
            else:
                allresult = np.concatenate((allresult, result), axis=0)

        assert len(allresult) == len(candids_feats)

        # select top candids index for regression model, sort by val in descending order
        top_idx = np.argpartition(-allresult, 5000)[:5000]

        top_candids_info = []
        top_candids_feats = []

        for idx in top_idx:
            candids_info[idx].append(allresult[idx])
            top_candids_info.append(candids_info[idx])
            top_candids_feats.append(candids_feats[idx])

        top_candids_feats = np.array(top_candids_feats, dtype=np.float32).reshape(-1, self.FragLen, 43)
        BlstmOut = self.sess.run([self.BLSTM], feed_dict={self.X: top_candids_feats, self.isTraining: False})
        BlstmOut = BlstmOut[0][:, -7:, :].reshape(-1, 7*128)

        target_contact = calc_contact_vec(self.TargetProt, target_locidx, self.FragLen)

        for i, candid in enumerate(top_candids_info):
            pdbidx = candid[1]
            locidx = candid[2]
            val = candid[3]

            candid_contact = calc_contact_vec(self.DB[pdbidx], locidx, self.FragLen)
            contact_simi = self.Contact.calculate_contact_simi(target_contact, candid_contact)

            reg_info = np.concatenate((np.array([self.FragLen, contact_simi, val]), BlstmOut[i]))
            reg_info = reg_info.astype(np.float32)

            candid.append(reg_info)

        return top_candids_info

    def __del__(self):
        del self.TargetProt, self.DB, self.FragLen, self.AllFeats, self.Contact

        self.sess.close()
        del self.sess
        tf.reset_default_graph()

        del self.X, self.Y_, self.BLSTM, self.isTraining


class Model_Regression(object):
    def __init__(self):
        tf.reset_default_graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False

        self.regsess = tf.Session(config=config)
        
        model_path = "models/reg/"
        tf.saved_model.loader.load(self.regsess, [tf.saved_model.tag_constants.SERVING], model_path)
     
        self.input_x_holder = self.regsess.graph.get_operation_by_name("inputs").outputs[0]
        self.predictions_holder = self.regsess.graph.get_operation_by_name("outputs").outputs[0]

    def predict(self, top_candids_info):
        reg_inputs = [x[4] for x in top_candids_info]
        reg_inputs = np.array(reg_inputs, dtype=np.float32).reshape(len(top_candids_info), 3+128*7)
        preds = self.regsess.run(self.predictions_holder, {self.input_x_holder: reg_inputs})

        for i, candid in enumerate(top_candids_info):
            candid[4] = preds[i]








