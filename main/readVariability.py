from config import config
# Load config parameters
locals().update(config)
import numpy as np
import pandas as pd
import sys
import pickle
from Statistics import *
from readNodes import *

#helper function to return T or F for True or False. This helps to cut down on fileName size
def bStr(boolStr):
    if boolStr:
        return "T"
    else:
        return "F"

def getpercentileID(l_pencentile, value):
    for i in range(len(l_pencentile) - 1):
        if value < l_pencentile[i + 1]:
            return int(i)
    return int(len(l_pencentile) - 1)


def getVariability(np_idtest, np_pred, np_pred_prob, d_propPerNodeAvg, d_propPerNodeStd, d_node_label, n_classes):
    l_cols = ['node', 'qid_avg','qid_std', 'correctness', 'loss']

    for i in range(n_classes):
        l_cols.append("error" + str(i))
    np_data = np.empty((0, len(l_cols)))
    l_avg_percentile = [np.percentile(d_propPerNodeAvg.values(), (i)*20, interpolation ='nearest') for i in range(5)]
    l_std_percentile = [np.percentile(d_propPerNodeStd.values(), (i)*20, interpolation ='nearest') for i in range(5)]

    print "l_avg_percentile:", l_avg_percentile
    print "l_std_percentile:", l_std_percentile


    for id in list(range(np_idtest.shape[0])):
        node = int(np_idtest[id][0])
        print d_node_label[node]
        true_class = int(d_node_label[node][0])
        predicted_class = int(np_pred[id][0])
        predicted_prob = np_pred_prob[id][true_class]
        loss = 1 - predicted_prob
        iscorrect = 0
        if true_class == predicted_class:
            iscorrect = 1
        error = 0
        l_error = [-999] * n_classes

        # print true_class, predicted_class, predicted_prob, np_pred_prob[id]
        l_error[true_class] = 1 - np_pred_prob[id][true_class]

        data = np.array([int(node), getpercentileID(l_avg_percentile, d_propPerNodeAvg[node]), getpercentileID(l_std_percentile, d_propPerNodeStd[node]), iscorrect, loss])
        data = np.reshape(data, (1, len(l_cols) - n_classes))

        prob = np.array(l_error)
        prob = np.reshape(prob, (1, n_classes))
        data = np.hstack((data, prob))
        np_data = np.vstack((np_data, data))


    index = ['Row' + str(i) for i in range(1, len(np_data) + 1)]

    df = pd.DataFrame(np_data, index=index, columns=l_cols)
    return df

def getNodesPercentiles(np_idtest, start, end, d_propPerNodeAvg, d_propPerNodeStd):

    l_cols = ['node', 'qid_avg', 'qid_std']
    np_data = np.empty((0, 3))
    l_avg_percentile = [np.percentile(d_propPerNodeAvg.values(), (i) * 20, interpolation='nearest') for i in range(5)]
    l_std_percentile = [np.percentile(d_propPerNodeStd.values(), (i) * 20, interpolation='nearest') for i in range(5)]

    print "l_avg_percentile:", l_avg_percentile
    print "l_std_percentile:", l_std_percentile

    for id in list(range(np_idtest.shape[0])):
        node = np_idtest[id][0]
        data = np.array([int(node), getpercentileID(l_avg_percentile, d_propPerNodeAvg[node]),
                         getpercentileID(l_std_percentile, d_propPerNodeStd[node])])
        data = np.reshape(data, (1, 3))
        np_data = np.vstack((np_data, data))

    index = ['Row' + str(i) for i in range(1, len(np_data) + 1)]

    df = pd.DataFrame(np_data, index=index, columns=l_cols)
    l_ids = list(df[(df['qid_avg'] >= start) & (df['qid_avg'] <= end)]['node'])
    l_ids = [int(i) for i in l_ids]
    return l_ids

def getnodeIDsfromProportion(start, end, dataset="facebook"):

    numFolds = 9
    l_test = []

    d_propPerNodeAvg, d_propPerNodeStd, _ = GetTimeStatisticsPerNode(
        str(sys.argv[1]) + "_filtered/preprocessed")

    for trial in range(0,10):
        f_data = dataFolder + dataset + "_filtered_trial_" + str(trial) + "_val_0.1.npy"
        nd_rest, nd_valid = np.load(f_data)
        folds = splitNodeFolds(nd_rest, numFolds)

        nd_train = []
        # take everything but last 2 folds
        # sum of last 2 folds is 0.2
        for k in range(0, numFolds - 2):
            nd_train += folds[k]
        nd_test = folds[-1] + folds[-2]
        l_test += nd_test
    np_idtest = np.reshape(np.array(l_test), (len(l_test), 1))
    l_data = getNodesPercentiles(np_idtest, start, end, d_propPerNodeAvg, d_propPerNodeStd)
    print "# num of nodes", len(set(l_data))
    return list(set(l_data))


def readJoels(fileName_option1):
    f = open(fileName_option1, 'r')
    lines = f.readlines()
    id_test = []
    pred = []
    pred_prob = []

    for line in lines[1:]:
        line = line.replace("\n", "")
        l_temps = line.split(",")
        id_test.append(int(l_temps[0]))
        pred_prob.append([1 - float(l_temps[1]), float(l_temps[1])])
        if float(l_temps[1]) > 0.5:
            pred.append(1)
        else:
            pred.append(0)
    return id_test, pred, pred_prob

def getbae_df(i, df_data, n_classes, targetmeasure):
    l_bae = []
    for k in range(n_classes):
        c_name = 'error' + str(k)
        l_bae.append(np.mean(
            list(df_data[(df_data[targetmeasure] == i) & (df_data[c_name] > -999)][
                c_name])))
        # print len(list(df_data[(df_data[targetmeasure] == i) & (df_data[c_name] > -999)][
        #          c_name]))
    print "# size: ", df_data.shape
    print df_data[0:10]
    return np.mean(l_bae)


# to print out scores using percentiles
def readConfigurations(dataName="facebook", option = 0, fileHead_option1 = "", fileTail_option1 = "", n_classes = 0):
    
    origName = "data_"+dataName+"_pool_0_pType_none_NNT_RNN_randInit_1_neighbSum_1_pNeighb_0_iterEpoch_1_randop_0_convs_0_convn_0_mpool_0_attsum_0_trial_0"


    # 0    dnType = "RNN"
    # 1    dnType = "CNN"
    # 2    dnType = "CollRNN"
    # 3    dnType = "AttentionRNN"
    # 4    dnType = "AttentionRNNAttr"

    NNTypes = ["RNN"]
    rInits = [1]
    neighbAvgs = [1]
    pNeighbs = [0]
    iterEpochs = [1]
    poolings = [1]
    randOps = [0]
    mpoolSizes = [4]
    convNums = [1]
    convSizes = [1]
    attsums = [0]
    trials = range(0, 10)
    configurations = []

    d_propPerNodeAvg, d_propPerNodeStd, d_node_label = GetTimeStatisticsPerNode(str(sys.argv[1]) + "_filtered/preprocessed")


    for nn in NNTypes:
        for attsum in attsums:
            for rinit in rInits:
                for mpool in mpoolSizes:
                    for convn in convNums:
                        for convs in convSizes:
                            for randop in randOps:            
                                for neighbavg in neighbAvgs:
                                    for pneighb in pNeighbs:
                                        for iterepoch in iterEpochs:
                                            for pool in poolings:
                                                np_idtest = np.empty((0, 1))
                                                np_pred = np.empty((0, 1))
                                                np_pred_prob = np.empty((0, 2))
                                                lossInput = []
                                                for i in trials:
                                                    newName = origName.replace("_pool_0", "_pool_"+str(pool)).replace("attsum_0", "attsum_"+str(attsum)).replace("NNT_RNN", "NNT_"+str(nn)).replace("mpool_0", "mpool_"+str(mpool)).replace("convn_0", "convn_"+str(convn)).replace("convs_0", "convs_"+str(convs)).replace("randop_0", "randop_"+str(randop)).replace("randInit_1", "randInit_"+str(rinit)).replace("neighbSum_1", "neighbSum_"+str(neighbavg)).replace("pNeighb_0", "pNeighb_"+str(pneighb)).replace("iterEpoch_1", "iterEpoch_"+str(iterepoch)).replace("trial_0", "trial_"+str(i))
#                                                     try:
#                                                         x = np.load(outputFolder+newName+".npy")
                                                    if option == 0: # hogun's
                                                        id_test, pred, pred_prob = pickle.load(open(predictFolder + newName, "rb"))
                                                    elif option == 1: # joel's
                                                        id_test, pred, pred_prob = readJoels(fileHead_option1 + str(i) + fileTail_option1)

                                                    id_test = np.reshape(id_test, (len(id_test), 1))
                                                    pred = np.reshape(pred, (len(pred), 1))
                                                    pred_prob = np.reshape(pred_prob, (len(pred_prob), 2))
                                                    np_idtest = np.vstack((np_idtest, id_test))
                                                    np_pred = np.vstack((np_pred, pred))
                                                    np_pred_prob = np.vstack((np_pred_prob, pred_prob))

                                                df_data = getVariability(np_idtest, np_pred, np_pred_prob, d_propPerNodeAvg, d_propPerNodeStd, d_node_label, n_classes)
                                                print newName
                                                print df_data.shape

                                                gb_avg = df_data.groupby(['qid_avg']).mean()
                                                gb_std = df_data.groupby(['qid_std']).mean()

                                                print gb_avg
                                                print gb_std

                                                text_avg_acc = ""
                                                text_avg_loss = ""
                                                text_std_acc = ""
                                                text_std_loss = ""
                                                text_avg_bae = ""
                                                text_std_bae = ""

                                                col_std = ""
                                                col_avg = ""



                                                for i in range(5):
                                                    col_avg += str(i) + "\t"
                                                    text_avg_acc += str(gb_avg.loc[i]['correctness']) + "\t"
                                                    text_avg_loss += str(gb_avg.loc[i]['loss']) + "\t"


                                                    text_avg_bae += str(getbae_df(float(i), df_data, n_classes,'qid_avg')) + "\t"

                                                    if i in list(gb_std.index):
                                                        col_std += str(i) + "\t"
                                                        text_std_acc += str(gb_std.loc[i]['correctness']) + "\t"
                                                        text_std_loss += str(gb_std.loc[i]['loss']) + "\t"
                                                        text_std_bae += str(getbae_df(i, df_data, n_classes, 'qid_std')) + "\t"
                                                    else:
                                                        col_std += str(i) + "\t"
                                                        text_std_acc += str(0.0) + "\t"
                                                        text_std_loss += str(0.0) + "\t"
                                                        text_std_bae += str(0.0) + "\t"

                                                        # text_std_loss += str(gb_std.loc[i]['loss']) + "\t"
                                                print "\n\n# avg_acc"
                                                print col_avg
                                                print text_avg_acc
                                                print "# avg_loss"
                                                print col_avg
                                                print text_avg_loss
                                                print "# avg_bae"
                                                print col_avg
                                                print text_avg_bae

                                                print "# std_acc"
                                                print col_std
                                                print text_std_acc
                                                print "# std_loss"
                                                print col_std
                                                print text_std_loss
                                                print "# std_bae"
                                                print col_std
                                                print text_std_bae


# generate trails using percentiles
# e.g
# starting percentile = 0 : 0 - 20%
# end  percentile = 1 : 20 - 40 %
def generateTrialsfromPercentiles(start, end):

    l_nodes = getnodeIDsfromProportion(start, end, dataset=sys.argv[1])
    trials = 10
    percentValidation = 0.1

    for i in range(0, trials):
        generateTrials(sys.argv[1] +"_selected_"+ str(start*20) + "_" + str((end+1)*20), trials, percentValidation, l_nodes)

def generateDatasets(dataName="facebook", start = 0, end = 0):

    l_nodes = getnodeIDsfromProportion(start, end, dataset=sys.argv[1])
    f_t_graph = dataFolder + dataName + "_filtered/preprocessed.p"
    f_t_output = dataFolder + dataName + "_selected_"+ str(start*20) + "_" + str((end+1)*20) +"/preprocessed.p"

    l_s_features, l_v_features, l_class_features, l_attr_features, n_users, n_attr = pickle.load(open(f_t_graph, "rb"))

    l_s_new_features = []
    l_v_new_features = []
    l_class_new_features = []
    l_attr_new_features = []

    for i in range(len(l_s_features)):
        if l_s_features[i] in l_nodes:
            l_s_new_features.append(l_s_features[i])
            l_v_new_features.append(l_v_features[i])
            l_class_new_features.append(l_class_features[i])
            l_attr_new_features.append(l_attr_features[i])


    len_s = len(l_s_features)

    l_v_features = l_v_new_features
    l_s_features = l_s_new_features

    l_class_features = l_class_new_features
    l_attr_features = l_attr_new_features

    while True:

        l_v_new_features = []
        l_s_new_features = []
        l_class_new_features = []
        l_attr_new_features = []

        l_sequences = []

        for row in range(len(l_v_features)):
            l_r_temp = []
            for graph in l_v_features[row]:
                l_temp = []
                for node in graph:
                    if node in l_s_features and node != l_s_features[row]:
                        l_temp.append(node)
                if len(l_temp) > 0:
                    l_r_temp.append(l_temp)
            if len(l_r_temp) > 0:
                l_sequences.append(len(l_r_temp))
                l_v_new_features.append(l_r_temp)
                l_s_new_features.append(l_s_features[row])
                l_class_new_features.append(l_class_features[row])
                l_attr_new_features.append(l_attr_features[row])

        l_v_features = l_v_new_features
        l_s_features = l_s_new_features

        l_class_features = l_class_new_features
        l_attr_features = l_attr_new_features
        print "avg length:", l_sequences

        if len_s == len(l_s_features):
            break
        else:
            len_s = len(l_s_features)
            print len_s
            print "--- doing iteration"

    print len(l_s_features)
    print len(l_v_features)
    print len(l_class_features)
    print len(l_attr_features)

    pickle.dump([l_s_features, l_v_features, l_class_features, l_attr_features, n_users, n_attr],
                open(f_t_output, "wb"))


if __name__ == '__main__':

    fname_head = "/Users/hogun/Downloads/imdb_binary1_newattr2_filtered/imdb_binary1_newattr2_filtered/imdb_binary1_newattr2_filtered_trial_"
    # fname_head = "/Users/hogun/Downloads/facebook_filtered/facebook_filtered/facebook_filtered_trial_"
    fname_tail = "_fold_6_LR.txt"
    # fname_tail = "_fold_6_PL-EM (CAL).txt"
    readConfigurations(dataName=sys.argv[1], option = 1, fileHead_option1 = fname_head, fileTail_option1 = fname_tail, n_classes = 2) # option 0 - hogun's option 1 - joel's



    # generateTrialsfromPercentiles(3, 4)
    # generateDatasets(dataName=sys.argv[1], start = 0, end = 1)
#
