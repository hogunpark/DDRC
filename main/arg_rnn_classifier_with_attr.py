'''
Author: Hogun park
Date: 10.11.2016
Goal: To run RNN for node classification

e.g.

use pooling and dntypeoption (0: without dist of neighbor labels, 1: with dist of the labels)
dataset abd n_classes are also should be modified

python arg_rnn_classifier_with_attr.py --dataset facebook --hiddenunits 32 --timedistunits 8 --maxpooling 4  --dntypeoption 0 --nclasses 2 --epochs 1 --parallel 0

'''

import pickle

import os

import numpy as np
from scipy import sparse
from keras.utils import np_utils
from keras.models import Sequential, Graph
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Convolution1D
from keras.layers.core import TimeDistributedDense, Masking, Reshape, Flatten, Activation, RepeatVector, Permute, Highway, Dropout, Merge, Lambda, TimeDistributedDense
from keras.layers import Input, Dense, Dropout, MaxPooling1D, MaxPooling2D

from keras.layers.wrappers import TimeDistributed
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras import backend as K
import theano.tensor as Th
import random
import math
import argparse
from sklearn import metrics

from nn_model import nn_model
from readNodes import *
from config import config
# Load config parameters
locals().update(config)

# from hogun.util.keras.util_keras import *


# get the maximum time stamp ID
def getmaxseq(l_v_features):
    maxseq = 0
    for row in l_v_features:
        if len(row) > maxseq:
            maxseq = len(row)
    return maxseq

def getbae(l_pred_prob, y_test, n_classes):
    l_baeall = []
    y_test = y_test.argmax(axis=-1)
    for c in range(n_classes):
        l_bae = []
        for j in range(len(y_test)):
            if y_test[j] == c:
                l_bae.append(1.0 - l_pred_prob[j][c])
        l_baeall.append(np.mean(l_bae))
    return np.mean(l_baeall)

def getzerooneloss(l_pred_prob, y_test):
    l_error = []
    y_test = y_test.argmax(axis=-1)

    for j in range(len(y_test)):
        trueclass = y_test[j]
        l_error.append(1.0 - l_pred_prob[j][trueclass])
    return np.mean(l_error)

# randomize edges across neighbors
def getrandedge(l_v_features):
    l_newv_features = []
    for row in l_v_features:
        l_sizes = []
        l_flatten = []
        l_new_v = []
        for graph in row:
            l_sizes.append(len(graph))
            l_flatten.append(list(graph))
        l_flatten = sum(l_flatten, [])
        random.shuffle(l_flatten, lambda: .7)

        index = 0
        for i in l_sizes:
            l_v = []
            for j in range(index, index + i):
                l_v.append(l_flatten[j])
            l_new_v.append(np.array(l_v))
            index += i
        l_newv_features.append(l_new_v)
    return l_newv_features

#helper function to return T or F for True or False. This helps to cut down on fileName size
def bStr(boolStr):
    if boolStr:
        return "T"
    else:
        return "F"

def reshape_features(l_v_features, l_attr_features, f_sparse_neighbor_features, maxseq, n_users, n_attr, rand_option = 0, b_parallel = 0, dntype_option = 0, normneighbor = 0):

    # to generate sparse examples for handling with very large graph
    if not preprocessed_dataset:

        l_nodes = []
        l_attr = []
        if rand_option == 2:
            l_v_features = getrandedge(l_v_features)

        for row in range(len(l_v_features)):
            l_all_graph = []
            l_aggregated_graph = np.array([0.] * (n_users + n_attr))
            for graph in l_v_features[row]:
                l = [0] * n_users
                n_count = 0
                for node in graph:
                    l[node] = 1
                    n_count += 1

                if normneighbor:
                    l = [float(ele)/n_count for ele in l]
                if dntype_option == 3 or dntype_option == 4:
                    l_all_graph.append(l)
                elif dntype_option == 5: # dnn
                    l_aggregated_graph +=  np.array(list(l_attr_features[row].toarray()[0]) + l)
                else:
                    l_all_graph.append(list(l_attr_features[row].toarray()[0]) + l)

            for z in range(maxseq - len(l_all_graph)):
                if dntype_option == 3 or dntype_option == 4:
                    l_all_graph.append([0.] * (n_users))
                elif dntype_option == 1:
                    l_all_graph.append([0] * (n_users + n_attr))
                elif dntype_option == 0:
                    if b_pooling:
                        l_all_graph.append([0.] * (n_users + n_attr))
                    else:
                        l_all_graph.append([-1.] * (n_users + n_attr))

            l_aggregated_graph = (l_aggregated_graph) / float(np.sum(l_aggregated_graph)) # for dntype_option = 5

            for i in range(len(l_aggregated_graph)):
                if l_aggregated_graph[i] > 0:
                    l_aggregated_graph[i] = 1


            if rand_option == 1:
                random.shuffle(l_all_graph, lambda: .3)
            if b_parallel:

                if dntype_option == 5:

                    l_all_graph = sparse.csr_matrix(l_aggregated_graph)  # to sparse matrix
                else:
                    l_all_graph = sparse.csr_matrix(l_all_graph)  # to sparse matrix

                l_attr.append(sparse.csr_matrix(list(l_attr_features[row].toarray()[0])))

            else:
                l_attr.append(list(l_attr_features[row].toarray()[0]))

                if dntype_option == 5:
                    l_all_graph = l_aggregated_graph

            l_nodes.append(l_all_graph)

            if row % 1000 == 0:
                print "--preprocessing ", row, " / ", len(l_v_features)
        data = l_nodes
        if dntype_option == 3 or dntype_option == 4:
            if not b_parallel:
                data = np.reshape(data, (len(data), maxseq, n_users))
            else:
                data = np.array(data)
        else:
            data = np.array(data)


        pickle.dump([data, l_attr], open(f_sparse_neighbor_features, "wb"))
    else:
        data, l_attr = pickle.load(open(f_sparse_neighbor_features, "rb"))


    return data, l_attr



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='RNN for time-sequence data')
    parser.add_argument('--dataset', type=str, help='Dataset file', required=True)
    parser.add_argument('--preprocesseddataset', type=int, default=0, help='Dataset file', required=False)
    parser.add_argument('--hiddenunits', type=int, help='Number of LSTM hidden units.',
                        default=256, required=False)
    parser.add_argument('--timedistunits', type=int, help='Number of TimeDist units.',
                        default=32, required=False)
    parser.add_argument('--maxpooling', type=int, help='Max Pooling.',
                        default=4, required=False)
    parser.add_argument('--convsize', type=int, help='Size of Conv Filter.',
                        default=4, required=False)
    parser.add_argument('--convnums', type=int, help='Number of Conv Filter.',
                        default=16, required=False)
    parser.add_argument('--randoption', type=int, help='0 - deafult, 1 - rand(graph), 2 - rand(edge)',
                        default=0, required=False)
    parser.add_argument('--dntypeoption', type=int, help='0 - rnn, 1 - cnn, 2 - collective version',
                        default=0, required=False)
    parser.add_argument('--batchsize', type=int, help='Number of sequences to process in a batch.',
                        default=5, required=False)
    parser.add_argument('--epochs', type=int, help='Number of epochs.',
                        default=30, required=False)
    parser.add_argument('--nclasses', type=int, help='Number of classes.',
                        default=2, required=True)
    parser.add_argument('--debug', type=int, help='0- debug false, 1- debug true.',
                        default=0, required=False)
    parser.add_argument('--parallel', type=int, help='0- parallel false, 1- parallel true.',
                        default=1, required=False)
    parser.add_argument('--pooling', type=int, help='0- pooling false, 1- pooling true.',
                        default=1, required=False)
    parser.add_argument('--whenupdate', type=int, help='0- update everyepoch, 1- update every intration',
                        default=0, required=False)
    parser.add_argument('--patience', type=int, help='num of patiences',
                        default=10, required=False)

    parser.add_argument('--pooltype', type=str, help='What type of pooling?',
                        default="none", required=False)

    #arguments for collective version experiments
    parser.add_argument('--rinit', type=int, help='Initialize randomly for collective inference',
                        default=1, required=False)
    parser.add_argument('--neighbsum', type=int, help='Add Neighbors to collective inference (1 - average sum)',
                        default=1, required=False)
    parser.add_argument('--perfectneighbors', type=int, help='Do we supply perfect neighbor labels to neighbor prediction/label sum? This helps to test given perfect label information, does the collective method perform well?',
                        default=0, required=False)
    parser.add_argument('--iterepoch', type=int, help='Do iterations equal epoch for collective inference?',
                        default=0, required=False)

    #trial information
    parser.add_argument('--trial', type=int, help='Which trial to run? -1 if you want to run all trials',
                        default=-1, required=False)
    parser.add_argument('--normneighbor', type=int, help='normlize neighbor nodes or not',
                        default=0, required=False)

    parser.add_argument('--attentionsum', type=int, help='if the sum of hidden nodes during attention is used ',
                        default=0, required=False)


    # 0 option for paralle does not work right now

    args = parser.parse_args()

    # init input params
    dataset = args.dataset
    hidden_units = args.hiddenunits
    timedist_units = args.timedistunits
    maxpool_size = args.maxpooling

    conv_size = args.convsize
    conv_nums = args.convnums

    rand_option = args.randoption
    dntype_option = args.dntypeoption

    batch_size = args.batchsize
    epochs = args.epochs
    debug = args.debug
    preprocessed_dataset = args.preprocesseddataset
    n_classes = args.nclasses

    b_parallel = args.parallel
    b_pooling = args.pooling

    patience = args.patience
    poolType = args.pooltype
    norm_neighbor = args.normneighbor

    #collective parameters
    rInit = args.rinit
    neighbSum = args.neighbsum
    perfectNeighbors = args.perfectneighbors
    iterEpoch = args.iterepoch
    trialInput = args.trial

    b_attention_sum = args.attentionsum

    numFolds = 9

    print args

    if dntype_option==0:
        dnType = "RNN"
    elif dntype_option==1:
        dnType = "CNN"
    elif dntype_option==2:
        dnType = "CollRNN"
    elif dntype_option==3:
        dnType = "AttentionRNN"
    elif dntype_option==4:
        dnType = "AttentionRNNAttr"
    elif dntype_option==5:
        dnType = "DNN"

    #create save_path_prefix
    save_path_prefix = "data_"+dataset+"_pool_"+str(b_pooling)+"_pType_"+str(poolType)+ \
        "_NNT_"+dnType+"_randInit_"+str(rInit)+"_neighbSum_"+str(neighbSum)+ \
        "_pNeighb_"+ str(perfectNeighbors)+"_iterEpoch_"+str(iterEpoch) + \
        "_randop_" + str(rand_option) + "_hunit_" + str(hidden_units) + "_tunit_" + str(timedist_units) + "_convs_" + str(conv_size) + "_convn_" + str(conv_nums) + "_mpool_" + str(maxpool_size)+ "_attsum_" + str(b_attention_sum)


    print save_path_prefix
    np.save(outputFolder+save_path_prefix+"_debugging", np.array(1))

    #if user supplied trial information
    if trialInput!=-1:
        save_path_prefix += "_trial_"+str(trialInput)



    # init model names to save the best model
    model_file = 'checkpoints/'
    model_file += save_path_prefix



    f_t_graph = dataFolder + dataset + "_filtered/preprocessed.p"
    f_sparse_neighbor_features = dataFolder + dataset + "_filtered/preprocessed_features_"+str(dntype_option)+".p"

    print "#file name:", f_t_graph


    l_s_features, l_v_features, l_class_features, l_attr_features, n_users, n_attr = pickle.load(open(f_t_graph, "rb"))

    n_users = int(n_users)
    n_attr = int(n_attr)

    maxseq = getmaxseq(l_v_features)
    print "# Max Time Seq: ", maxseq

    data, raw_attr = reshape_features(l_v_features, l_attr_features, f_sparse_neighbor_features, maxseq, n_users, n_attr, rand_option, b_parallel, dntype_option, norm_neighbor)


    if dntype_option != 3 or dntype_option != 4:
        feat_size = n_users + n_attr
    else:
        feat_size = n_users

    print np.array(raw_attr).shape
    if b_parallel: raw_attr= np.array(raw_attr)
    else: raw_attr = np.reshape(np.array(raw_attr), (len(raw_attr), n_attr))


    raw_labels = np.array(l_class_features)
    raw_labels = np.reshape(raw_labels, (len(l_class_features), 1))
    labels = np_utils.to_categorical(raw_labels, n_classes)

    x_all = data
    y_all = labels
    id_all = np.array(l_s_features)

    n_data = len(l_s_features)

    for c in range(n_classes):
        print "# Class ID (", c, ")", l_class_features.count(c)

    l_accuracy = []
    l_bae = []
    l_zerooneloss = []

    for trial in range(0,20):

        print "####################################"
        print "############## TRIAL ", trial, "######"
        print "####################################"

        f_data = dataFolder + dataset + "_filtered_trial_" + str(trial) + "_val_0.1.npy"
        nd_rest, nd_valid = np.load(f_data)
        folds = splitNodeFolds(nd_rest, numFolds)

        nd_train = []
        # take everything but last 2 folds
        # sum of last 2 folds is 0.2
        for k in range(0, numFolds - 2):
            nd_train += folds[k]
        nd_test = folds[-1] + folds[-2]


        f_log = open(model_file, 'w') # clear the previous one
        f_log.write("\n")
        f_log.close()

        idx_train = []
        idx_test = []
        idx_valid = []
        idx_all = []


        d_idx_sid = {}
        d_sid_idx = {}

        for k in range(len(l_s_features)):
            sid = l_s_features[k]
            d_sid_idx[sid] = k
            d_idx_sid[k] = sid

        for k in range(len(l_s_features)):
            sid = l_s_features[k]
            original_idx = d_sid_idx[sid]
            idx_all.append(original_idx)
            if sid in nd_train: idx_train.append(original_idx)
            if sid in nd_test: idx_test.append(original_idx)
            if sid in nd_valid: idx_valid.append(original_idx)

        idx_unlabeled = idx_valid + idx_test

        #if user provides trialInput
        #and not equal to current iteration, then skip
        if trialInput!=-1 and trialInput!=trial:
            continue


        x_train = np.array([data[k] for k in idx_train])
        y_train = np.array([labels[k] for k in idx_train])
        attr_train = np.array([raw_attr[k] for k in idx_train])
        id_train = np.array([l_s_features[k] for k in idx_train])

        x_valid = np.array([data[k] for k in idx_valid])
        y_valid = np.array([labels[k] for k in idx_valid])
        attr_valid = np.array([raw_attr[k] for k in idx_valid])
        id_valid = np.array([l_s_features[k] for k in idx_valid])

        x_test = np.array([data[k] for k in idx_test])
        y_test = np.array([labels[k] for k in idx_test])
        attr_test = np.array([raw_attr[k] for k in idx_test])
        id_test = np.array([l_s_features[k] for k in idx_test])

        x_unlabeled = np.array([data[k] for k in idx_unlabeled])
        y_unlabeled = np.array([labels[k] for k in idx_unlabeled])
        attr_unlabeled = np.array([raw_attr[k] for k in idx_unlabeled])
        id_unlabeled = np.array([l_s_features[k] for k in idx_unlabeled])


        print "# y_all shape:", labels.shape
        print "# x_all shape:", data.shape
        print "# y_train shape:", y_train.shape
        print "# maxseq: ", maxseq
        print "# n users: ", n_users

        # build model
        i_model = nn_model(maxseq, n_users, n_attr, n_classes, b_parallel, neighbSum, perfectNeighbors, rInit, patience)
        i_model.build_model(dntype_option, hidden_units, timedist_units, maxpool_size, conv_nums, conv_size, b_pooling, b_attention_sum)

        # Setup callbacks
        callbacks = [
            ModelCheckpoint(model_file, monitor='val_acc',
                            verbose=0, save_best_only=True, mode='max'),
            EarlyStopping(monitor='val_acc', patience=patience, verbose=3, mode='max'),
        ]
        # Train
        i_model.train_model(model_file, x_train, y_train, attr_train, id_train, x_valid, y_valid, attr_valid, x_unlabeled, y_unlabeled, attr_unlabeled, id_unlabeled, y_all, id_all, callbacks, batch_size, epochs)
        # Test
        score, acc= i_model.evaluate_model(x_test, y_test, attr_test, model_file, batch_size)
        print "# accuracy:", acc
        # Evaluate
        l_pred = i_model.predict(x_test, attr_test, model_file, batch_size)
        l_pred_prob = i_model.predict_prob(x_test, attr_test, model_file, batch_size)
        bae = getbae(l_pred_prob, y_test, n_classes)
        zerooneloss = getzerooneloss(l_pred_prob, y_test)
        print "# bae: ", bae
        print "# zerooneloss: ", zerooneloss

        l_accuracy.append(acc)
        l_bae.append(bae)
        l_zerooneloss.append(zerooneloss)

        pickle.dump([np.array(id_test), np.array(l_pred), np.array(l_pred_prob)], open(predictFolder + save_path_prefix, "wb"))


    print 'Avg accuracy: ', np.mean(l_accuracy), np.std(l_accuracy)
    print 'Avg bae: ', np.mean(l_bae), np.std(l_bae)
    print 'Avg zero-one loss: ', np.mean(l_zerooneloss), np.std(l_zerooneloss)

