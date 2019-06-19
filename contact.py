# -*- coding: utf-8 -*-

from __future__ import division, print_function

import copy
import sys

import numpy as np
import tensorflow as tf


class Contact(object):
    '''
        target: the name of protein target
        ss: an N*3 array which shows the predicted probablity of H E C of each residue.
    '''

    def __init__(self, targetprot):
        self.TargetProt = targetprot
        ccmpredpath = "examples/inputs/{}.mat".format(self.TargetProt.name)
        # N x N
        ccmpred = np.loadtxt(ccmpredpath)
        self.ccmpred = (ccmpred+ccmpred.T)/2.
        
        # N x 3, H E C
        self.ss = self.TargetProt.ss.reshape(-1, 3)
        self.length = len(self.ss)

    def pred_by_local_contact_model(self, inputs):
        tf.reset_default_graph()

        config = tf.ConfigProto()
        graph_local = tf.Graph()
        local_sess = tf.Session(config=config, graph=graph_local)
        
        with local_sess.as_default():
            with local_sess.graph.as_default():
                graph_def_local = tf.GraphDef()
                with tf.gfile.FastGFile('models/contact/local.pb', 'rb') as f:
                    graph_def_local.ParseFromString(f.read())
                    tf.import_graph_def(graph_def_local, name='')
            
                    o = graph_local.get_tensor_by_name('Output:0')
                    x = graph_local.get_tensor_by_name('x:0')

                    pred = local_sess.run(o, feed_dict={x:inputs})

                    return pred

    def pred_by_nonlocal_contact_model(self, inputs):
        tf.reset_default_graph()

        config = tf.ConfigProto()
        graph_nonlocal = tf.Graph()
        nonlocal_sess = tf.Session(config=config, graph=graph_nonlocal)
        
        with nonlocal_sess.as_default():
            with nonlocal_sess.graph.as_default():
                graph_def_nonlocal = tf.GraphDef()
                with tf.gfile.FastGFile('models/contact/nonlocal.pb', 'rb') as f:
                    graph_def_nonlocal.ParseFromString(f.read())
                    tf.import_graph_def(graph_def_nonlocal, name='')
            
                    o = graph_nonlocal.get_tensor_by_name('Output:0')
                    x = graph_nonlocal.get_tensor_by_name('x:0')

                    pred = nonlocal_sess.run(o, feed_dict={x:inputs})

                    return pred

    def pred_contact_raw(self, mode):
        '''
        mode: local/ non-local contact prediction mode
        '''

        deepcnf2d = np.concatenate((self.ss[np.newaxis, :, :] * np.ones((self.length, 1, 1)),
                                    self.ss[:, np.newaxis, :] * np.ones((1, self.length, 1))),
                                   axis=2)
        pos = np.abs(np.arange(self.length)[np.newaxis] - np.arange(self.length)[:, np.newaxis])
        feature2d = np.concatenate((self.ccmpred[:, :, np.newaxis], deepcnf2d, pos[:, :, np.newaxis]), axis=2)
        
        if mode == "nonlocal":
            pred = self.pred_by_nonlocal_contact_model(feature2d[np.newaxis])

        elif mode == "local":
            pred = self.pred_by_local_contact_model(feature2d[np.newaxis])
        
        else:
            print ("Uncognized model label!")
            sys.exit()

        r = pred.squeeze()
        r = (r+r.T)/2.

        return r

    def pred_contact(self):
        a = self.pred_contact_raw("nonlocal")
        b = self.pred_contact_raw("local")

        assert len(a) == len(b)
        assert len(a) == self.length

        contact = np.zeros((self.length, self.length))

        for i in range(self.length):
            for j in range(self.length):
                if abs(i-j) < 6:
                    contact[i][j] = b[i][j]
                else:
                    contact[i][j] = a[i][j]

        return contact

    def calculate_contact_simi(self, target_vec, candid_vec):
        simi = np.linalg.norm(target_vec - candid_vec) / np.sqrt(len(target_vec))
        return simi


def selec_by_posi(vec):
    res = []
    for i, row in enumerate(vec):
        for j, ele in enumerate(row):
            if i-j > 0:
                res.append(ele)
    return np.array(res)


def calc_contact_vec(chain, loc, fraglen):
    vec = chain.contact_vec[loc:loc+fraglen, loc:loc+fraglen]
    vec = selec_by_posi(vec)
    return vec
