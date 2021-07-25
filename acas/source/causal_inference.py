#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-06-12 16:27:19
# @Author  : Sun Bing

import numpy as np
from keras import backend as K

from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from keras.layers import UpSampling2D, Cropping2D
from keras.layers import Input
from keras import Model
import pyswarms as ps

import utils_backdoor

from decimal import Decimal

import os
import sys
import time
from keras.preprocessing import image

##############################
#        PARAMETERS          #
##############################

RESULT_DIR = '../results'  # directory for storing results
IMG_FILENAME_TEMPLATE = 'mnist_visualize_%s_label_%d.png'  # image filename template for visualization results

# input size
IMG_ROWS = 28
IMG_COLS = 28
IMG_COLOR = 1
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, IMG_COLOR)
MASK_SHAPE = (IMG_ROWS, IMG_COLS)

NUM_CLASSES = 10  # total number of classes in the model

CALSAL_STEP = 4

TEST_ONLY = 1

class causal_analyzer:

    BATCH_SIZE = 32
    # verbose level, 0, 1 or 2
    VERBOSE = 1
    # dir to save intermediate masks
    TMP_DIR = 'tmp'

    SPLIT_LAYER = 6
    REP_N = 5

    def __init__(self, model, dd_generator, cex_generator, input_shape,
                 mini_batch, batch_size=BATCH_SIZE, verbose=VERBOSE,
                 rep_n=REP_N):


        self.model = model
        self.input_shape = input_shape
        self.dd_gen = dd_generator
        self.cex_gen = cex_generator
        self.steps = 1  #steps
        self.mini_batch = mini_batch
        self.batch_size = batch_size
        self.verbose = verbose

        self.rep_n = rep_n       # number of neurons to repair
        self.r_weight = None
        self.target = 0
        self.alpha = 0.3

        # split the model for causal inervention
        '''
        self.model1 = Model(inputs=self.model.inputs, outputs=self.model.layers[5].output)
        model2_input = Input(self.model.layers[6].input_shape[1:])
        self.model12 = model2_input
        for layer in self.model.layers[6:]:
            self.model12 = layer(self.model12)
        self.model12 = Model(inputs=model2_input, outputs=self.model12)
        '''
        self.model1, self.model2 = self.split_keras_model(self.model, self.SPLIT_LAYER)

        pass

    def split_keras_model(self, lmodel, index):

        model1 = Model(inputs=lmodel.inputs, outputs=lmodel.layers[index - 1].output)
        model2_input = Input(lmodel.layers[index].input_shape[1:])
        model2 = model2_input
        for layer in lmodel.layers[index:]:
            model2 = layer(model2)
        model2 = Model(inputs=model2_input, outputs=model2)

        return (model1, model2)


    def analyze(self, dd_gen, cex_gen):
        #'''
        # find hidden range
        for step in range(self.steps):
            min = []
            min_p = []
            max = []
            max_p = []
            #self.mini_batch = 2
            for idx in range(self.mini_batch):
                X_batch = np.squeeze(np.squeeze(dd_gen.next(), axis=2), axis=2)
                X_batch_cex = np.squeeze(np.squeeze(cex_gen.next(), axis=2), axis=2)
                min_i, max_i = self.get_h_range(X_batch)
                min.append(min_i)
                max.append(max_i)

                min_i, max_i = self.get_h_range(X_batch_cex)
                min_p.append(min_i)
                max_p.append(max_i)

                p_prediction = self.model.predict(X_batch_cex)
                ori_predict = self.model.predict(X_batch)
                np.savetxt("../results/p_prediction.txt", p_prediction, fmt="%s")
                np.savetxt("../results/ori_predict.txt", ori_predict, fmt="%s")
                predict = np.argmin(p_prediction, axis=1)
                ori_predict = np.argmin(ori_predict, axis=1)

            min = np.min(np.array(min), axis=0)
            max = np.max(np.array(max), axis=0)

            min_p = np.min(np.array(min_p), axis=0)
            max_p = np.max(np.array(max_p), axis=0)
        #'''
        # loop start

        for step in range(self.steps):
            #'''
            ie_batch = []
            ie_batch1 = []
            #self.mini_batch = 2
            for idx in range(self.mini_batch):
                X_batch = np.squeeze(np.squeeze(dd_gen.next(), axis=2), axis=2)

                #X_batch_perturbed = self.get_perturbed_input(X_batch)

                # find hidden neuron interval

                # find
                #ie_batch.append(self.get_ie_do_h(X_batch, np.minimum(min_p, min), np.maximum(max_p, max)))
                ie0, ie1 = self.get_tie_do_h(X_batch, self.target, np.minimum(min_p, min), np.maximum(max_p, max))
                ie_batch.append(ie0)
                ie_batch1.append(ie1)

            ie_mean = np.mean(np.array(ie_batch),axis=0)
            ie_mean1 = np.mean(np.array(ie_batch1), axis=0)

            np.savetxt("../results/ori.txt", ie_mean, fmt="%s")
            np.savetxt("../results/ori1.txt", ie_mean1, fmt="%s")
            #return
            # ie_mean dim: 512 * 43
            # find tarted class: diff of each column
            col_diff = np.max(ie_mean, axis=0) - np.min(ie_mean, axis=0)
            col_diff = np.transpose([np.arange(len(col_diff)), col_diff])
            ind = np.argsort(col_diff[:, 1])[::-1]
            col_diff = col_diff[ind]

            np.savetxt("../results/col_diff.txt", col_diff, fmt="%s")

            col_diff1 = np.max(ie_mean1, axis=0) - np.min(ie_mean1, axis=0)
            col_diff1 = np.transpose([np.arange(len(col_diff1)), col_diff1])
            ind = np.argsort(col_diff1[:, 1])[::-1]
            col_diff1 = col_diff1[ind]

            np.savetxt("../results/col_diff1.txt", col_diff1, fmt="%s")

            row_diff = np.max(ie_mean, axis=1) - np.min(ie_mean, axis=1)
            row_diff = np.transpose([np.arange(len(row_diff)), row_diff])
            ind = np.argsort(row_diff[:, 1])[::-1]
            row_diff = row_diff[ind]

            np.savetxt("../results/row_diff.txt", row_diff, fmt="%s")

            row_diff1 = np.max(ie_mean1, axis=1) - np.min(ie_mean1, axis=1)
            row_diff1 = np.transpose([np.arange(len(row_diff1)), row_diff1])
            ind = np.argsort(row_diff1[:, 1])[::-1]
            row_diff1 = row_diff1[ind]

            np.savetxt("../results/row_diff1.txt", row_diff1, fmt="%s")
            #'''
            # row_diff contains sensitive neurons: top self.rep_n
            # index
            self.rep_index = []
            result, acc = self.pso_test([], self.target)
            print("before repair: attack SR: {}, BE acc: {}".format(result, acc))

            rep_idx = row_diff[:,:1][:self.rep_n,:].tolist()
            rep1_idx = row_diff1[:,:1][:self.rep_n,:].tolist()

            result_list = rep_idx
            result_list.extend(x for x in rep1_idx if x not in rep_idx)
            result_list = np.array(result_list)

            self.rep_index = result_list[:self.rep_n]

            self.rep_index = row_diff[:,:1][:self.rep_n,:]
            print("repair index: {}".format(self.rep_index.T))

            self.repair()

            #self.rep_index = [461, 395, 491, 404, 219]
            #self.r_weight = [-0.13325777,  0.08095828, -0.80547224, -0.59831971, -0.23067632]

            result, acc = self.pso_test(self.r_weight, self.target)
            print("after repair: attack SR: {}, BE acc: {}".format(result, acc))

    pass

    # return
    def get_ie_do_h(self, x, min, max):
        pre_layer5 = self.model1.predict(x)
        l_shape = pre_layer5.shape
        ie = []

        hidden_min = min.reshape(-1)
        hidden_max = max.reshape(-1)
        num_step = CALSAL_STEP

        _pre_layer5 = np.reshape(pre_layer5, (len(pre_layer5), -1))

        for i in range (len(_pre_layer5[0])):
            ie_i = []
            for h_val in np.linspace(hidden_min[i], hidden_max[i], num_step):
                do_hidden = _pre_layer5.copy()
                do_hidden[:, i] = h_val
                pre_final = self.model2.predict(do_hidden.reshape(l_shape))
                ie_i.append(np.mean(pre_final,axis=0))
            ie.append(np.mean(np.array(ie_i),axis=0))
        return np.array(ie)

    # get ie of targeted class
    def get_tie_do_h(self, x, t_dix, min, max):
        pre_layer5 = self.model1.predict(x)
        l_shape = pre_layer5.shape
        ie = []
        ie2 = []

        hidden_min = min.reshape(-1)
        hidden_max = max.reshape(-1)
        num_step = CALSAL_STEP

        _pre_layer5 = np.reshape(pre_layer5, (len(pre_layer5), -1))

        for i in range (len(_pre_layer5[0])):
            ie_i = []
            ie2_i = []
            for h_val in np.linspace(hidden_min[i], hidden_max[i], num_step):
                do_hidden = _pre_layer5.copy()
                do_hidden[:, i] = h_val
                pre_final = self.model2.predict(do_hidden.reshape(l_shape))
                ie_i.append(np.mean(pre_final,axis=0)[t_dix])
                ie2_i.append(np.mean(pre_final,axis=0)[1])
            ie.append(np.array(ie_i))
            ie2.append(np.array(ie2_i))
        return np.array(ie), np.array(ie2)

    # return
    def get_die_do_h(self, x, x_p, min, max):
        pre_layer5 = self.model1.predict(x)
        pre_layer5_p = self.model1.predict(x_p)

        ie = []

        hidden_min = min
        hidden_max = max
        num_step = 16

        for i in range (len(pre_layer5[0])):
            ie_i = []
            for h_val in np.linspace(hidden_min[i], hidden_max[i], num_step):
                do_hidden = pre_layer5.copy()
                do_hidden[:, i] = h_val
                pre_final = self.model2.predict(do_hidden)
                pre_final_ori = self.model2.predict(pre_layer5_p)
                ie_i.append(np.mean(np.absolute(pre_final - pre_final_ori),axis=0))
            ie.append(np.mean(np.array(ie_i),axis=0))
        return np.array(ie)

    # return
    def get_final(self, x, x_p, min, max):
        return np.mean(self.model.predict(x),axis=0)

    def get_h_range(self, x):
        pre_layer5 = self.model1.predict(x)

        max = np.max(pre_layer5,axis=0)
        min = np.min(pre_layer5, axis=0)

        return min, max

    def repair(self):
        # repair
        print('Start reparing...')
        options = {'c1': 0.41, 'c2': 0.41, 'w': 0.8}
        #'''# original
        optimizer = ps.single.GlobalBestPSO(n_particles=20, dimensions=self.rep_n, options=options,
                                            bounds=([[-10.0] * self.rep_n, [10.0] * self.rep_n]),
                                            init_pos=np.ones((20, self.rep_n), dtype=float), ftol=1e-3,
                                            ftol_iter=10)
        #'''

        # Perform optimization
        best_cost, best_pos = optimizer.optimize(self.pso_fitness_func, iters=100)

        # Obtain the cost history
        # print(optimizer.cost_history)
        # Obtain the position history
        # print(optimizer.pos_history)
        # Obtain the velocity history
        # print(optimizer.velocity_history)
        #print('neuron to repair: {} at layter: {}'.format(self.r_neuron, self.r_layer))
        #print('best cost: {}'.format(best_cost))
        #print('best pos: {}'.format(best_pos))

        self.r_weight = best_pos

        return best_pos

    # optimization target perturbed sample has the same label as clean sample
    def pso_fitness_func(self, weight):

        result = []
        for i in range (0, int(len(weight))):
            r_weight =  weight[i]

            cost = self.pso_test_rep(r_weight)

            #print('cost: {}'.format(cost))

            result.append(cost)

        #print(result)

        return result

    def pso_test_rep(self, r_weight):
        result = 0.0
        correct = 0.0
        tot_count = 0
        # per particle
        for idx in range(self.mini_batch):
            X_batch = np.squeeze(np.squeeze(self.dd_gen.next(), axis=2), axis=2)
            X_batch_perturbed = np.squeeze(np.squeeze(self.cex_gen.next(), axis=2), axis=2)

            o_prediction = self.model1.predict(X_batch)
            p_prediction = self.model1.predict(X_batch_perturbed)

            _p_prediction = np.reshape(p_prediction, (len(p_prediction), -1))
            _o_prediction = np.reshape(o_prediction, (len(o_prediction), -1))

            l_shape = p_prediction.shape

            do_hidden = _p_prediction.copy()
            o_hidden = _o_prediction.copy()

            for i in range(0, len(self.rep_index)):
                rep_idx = int(self.rep_index[i])
                do_hidden[:, rep_idx] = (r_weight[i]) * _p_prediction[:, rep_idx]
                o_hidden[:, rep_idx] = (r_weight[i]) * _o_prediction[:, rep_idx]

            p_prediction = self.model2.predict(do_hidden.reshape(l_shape))
            o_prediction = self.model2.predict(o_hidden.reshape(l_shape))

            p_prediction = np.argmin(p_prediction, axis=1)
            o_prediction = np.argmin(o_prediction, axis=1)

            attack_success = len(o_prediction) - np.sum(p_prediction == 0 * np.ones(p_prediction.shape)) - np.sum(
                p_prediction == 1 * np.ones(p_prediction.shape))
            result = result + attack_success

            o_correct = np.sum(o_prediction == 0 * np.ones(o_prediction.shape)) + np.sum(
                o_prediction == 1 * np.ones(o_prediction.shape))
            correct = correct + o_correct
            tot_count = tot_count + len(o_prediction)

        result = result / tot_count
        correct = correct / tot_count
        cost = (1.0 - self.alpha) * result - self.alpha * correct
        return cost

    def pso_test(self, r_weight, target):
        result = 0.0
        correct = 0.0
        tot_count = 0
        if len(self.rep_index) != 0:

            # per particle
            for idx in range(self.mini_batch):
                X_batch = np.squeeze(np.squeeze(self.dd_gen.next(), axis=2), axis=2)
                X_batch_perturbed = np.squeeze(np.squeeze(self.cex_gen.next(), axis=2), axis=2)

                o_prediction = self.model1.predict(X_batch)
                p_prediction = self.model1.predict(X_batch_perturbed)

                _p_prediction = np.reshape(p_prediction, (len(p_prediction), -1))
                _o_prediction = np.reshape(o_prediction, (len(o_prediction), -1))

                l_shape = p_prediction.shape

                do_hidden = _p_prediction.copy()
                o_hidden = _o_prediction.copy()

                for i in range (0, len(self.rep_index)):
                    rep_idx = int(self.rep_index[i])
                    do_hidden[:, rep_idx] = (r_weight[i]) * _p_prediction[:, rep_idx]
                    o_hidden[:, rep_idx] = (r_weight[i]) * _o_prediction[:, rep_idx]

                p_prediction = self.model2.predict(do_hidden.reshape(l_shape))
                o_prediction = self.model2.predict(o_hidden.reshape(l_shape))

                p_prediction = np.argmin(p_prediction, axis=1)
                o_prediction = np.argmin(o_prediction, axis=1)

                attack_success = len(o_prediction) - np.sum(p_prediction == 0 * np.ones(p_prediction.shape)) - np.sum(p_prediction == 1 * np.ones(p_prediction.shape))
                result = result + attack_success

                o_correct = np.sum(o_prediction == 0 * np.ones(o_prediction.shape)) + np.sum(o_prediction == 1 * np.ones(o_prediction.shape))
                correct = correct + o_correct
                tot_count = tot_count + len(o_prediction)

            result = result / tot_count
            correct = correct / tot_count
        else:
            # per particle
            for idx in range(self.mini_batch):
                X_batch = np.squeeze(np.squeeze(self.dd_gen.next(), axis=2), axis=2)
                X_batch_perturbed = np.squeeze(np.squeeze(self.cex_gen.next(), axis=2), axis=2)

                o_prediction = np.argmin(self.model.predict(X_batch), axis=1)
                p_prediction = np.argmin(self.model.predict(X_batch_perturbed), axis=1)

                attack_success = len(o_prediction) - np.sum(p_prediction == 0 * np.ones(p_prediction.shape)) - np.sum(p_prediction == 1 * np.ones(p_prediction.shape))
                result = result + attack_success

                o_correct = np.sum(o_prediction == 0 * np.ones(o_prediction.shape)) + np.sum(o_prediction == 1 * np.ones(o_prediction.shape))
                correct = correct + o_correct
                tot_count = tot_count + len(o_prediction)
            result = result / tot_count
            correct = correct / tot_count
        return result, correct
