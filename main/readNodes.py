##Use these utilities to ensure we are using the same nodes for each method we compare to##

from random import shuffle
import sys
import random
import numpy as np
import os
import math
import networkx as nx
import pickle
# Load config parameters
from config import config
from Statistics import *
from readVariability import *
locals().update(config)


#read a hogun network :)
def readHogun(dataFolder, fName):
    f_t_graph  = dataFolder+fName+"/preprocessed.p"
    x = pickle.load( open( f_t_graph, "rb" ) )
    nodes = x[0] ; neighborTimeSeries = x[1] ; labels = x[2] ; features = x[3] ;
    G = nx.Graph()
    for i in range(0, len(nodes)):
        x = nodes[i]
        #if x == 4868:
        #    print(x)
        #    print(neighborTimeSeries[i])
        G.add_node(x)
        G.node[x]['label'] = [labels[i]]
        G.node[x]['dynamic_label'] = [float(labels[i])]
        if type(features[i]) is np.ndarray:
            G.node[x]['attr'] = features[i].tolist()
        else:
            G.node[x]['attr'] = features[i].toarray().tolist()[0]
        G.node[x]['neighbors'] = neighborTimeSeries[i]

    return G

def convertToJohn(fName):
    G = readHogun(dataFolder, fName)
    for node in G.nodes():
        neighbors = {}
        for neighborTime in G.node[node]['neighbors']:
            for neighbID in neighborTime:
                neighbors[neighbID]=1
        for key in neighbors:
            #remove to test if all nodes in edge set are in node set
            #if key in G.nodes():
            #1250-1249
            G.add_edge(node, key)

    fAttr = open("data/"+fName+".attr", 'w')
    fLab = open("data/"+fName+".lab", 'w')
    fEdges = open("data/"+fName+".edges", 'w')
    for node in G.nodes():
        fAttr.write(str(node))
        fLab.write(str(node))
        #write attrs
        #print(node)
        #print(G.node[node])
        for attrVal in G.node[node]['attr']:
            fAttr.write("::"+str(attrVal))
        #write labs
        for lab in G.node[node]['label']:
            fLab.write("::"+str(lab))
        fAttr.write("\n")
        fLab.write("\n")
    fAttr.close()
    fLab.close()

    edgesHash = {}
    for edge in G.edges():
        candEdge = [str(node) for node in sorted(edge)]
        edgesHash["-".join(candEdge)]=1
    for edge in edgesHash:
        nodePair = edge.split("-")
        fEdges.write(nodePair[0]+"::"+nodePair[1]+"\n")
    fEdges.close()

    return G

#split "nodes" into two lists. An individual node has a "percent" chance of appearing in the first list
def splitNodes(nodes,trainPercent, valPercent):
    group1=[]
    group2=[]
    group3=[]
    shuffle(nodes)
    valNext=trainPercent+valPercent
    for node in nodes:
        rand=random.random()
        if rand < trainPercent:
            group1.append(node)
        elif rand> trainPercent and rand< valNext:
            group2.append(node)
        elif rand>valNext:
            group3.append(node)
            
    return (group1,group2, group3)


#use to split test nodes
def splitNodeFolds(nodes, numFolds):
    numFolds = int(numFolds)
    cuts=int(math.ceil(float(len(nodes))/numFolds))
    folds = []
    for i in xrange(0, len(nodes), cuts):
        folds.append(nodes[i:min(i+cuts, len(nodes))])
        
    return folds

#saves each trials
#ONLY GENERATE ONCE AND USE ACROSS ALL ALGORITHMS
def generateTrialsbyTGraph(fName, trials, percentValidation):
    print(fName)
    G = readHogun(dataFolder, fName)
    for i in range(10, 10+trials):
        #get random partitioning
        (validationNodes, testNodes, rest) = splitNodes(G.nodes(),percentValidation, 0.0)
        #shuffle(validationNodes)
        #shuffle(testNodes)
        shuffle(rest)
        npArr = np.array([rest, validationNodes], dtype='object')
        np.save(dataFolder+fName+"_trial_"+str(i)+"_val_"+str(percentValidation), npArr)

# saves each trials
# ONLY GENERATE ONCE AND USE ACROSS ALL ALGORITHMS
def generateTrialsbyNode(fName, trials, percentValidation, l_nodes):
    print(fName)

    for i in range(0, trials):
        # get random partitioning
        (validationNodes, testNodes, rest) = splitNodes(l_nodes, percentValidation, 0.0)
        # shuffle(validationNodes)
        # shuffle(testNodes)
        shuffle(rest)
        npArr = np.array([rest, validationNodes], dtype='object')
        np.save(dataFolder + fName + "_trial_" + str(i) + "_val_" + str(percentValidation), npArr)

#need to pass in trial # and percentValidation
def readTrial(dataFolder, fName, i, percentValidation):
    trial = np.load(dataFolder+fName+"_trial_"+str(i)+"_val_"+str(percentValidation)+".npy")
    rest, valNodes=trial.tolist()
    return (rest, valNodes)


def outputProportions(fName):
    #G = readDataset("data/", fName)
    dataFolder = "../../experiments/data/"
    G = readHogun(dataFolder, fName)
    pos = 0
    neg = 0
    lenNodes = 0
    lenEdges = len(G.edges())
    for node in G.nodes():
        degree = len(G.neighbors(node))
        if degree>-1:
            lenNodes+=1
            if G.node[node]['label'][0] == 1:
                pos+=1
            else:
                neg+=1
    print(fName)
    print("numNodes:" + str(lenNodes))
    print("lenEdges:"+str(lenEdges))
    print("pos: "+str(float(pos)/lenNodes))
    print("neg: "+str(float(neg)/lenNodes))

def exampleRead():
    fName = "preprocessed_facebook"
    i = 0

    ##IMPORTANT##
    #Keep percentValidation and numFolds static
    #This is so that we read same nodes
    percentValidation = 0.1
    #0.9 is left of the whole network
    #this yields 0.1 per fold
    numFolds = 9

    rest, validationNodes= readTrial(dataFolder, fName, i, percentValidation)
    folds= splitNodeFolds(rest, numFolds)

    trainNodes=[] 
    #take everything but last 2 folds
    #sum of last 2 folds is 0.2
    for k in range(0, numFolds-2):
        trainNodes+=folds[k]
    testNodes = folds[-1]+folds[-2]
    print(len(trainNodes+testNodes))   

def getloss(id_test, pred, pred_prob, d_node_label):
    l_error = []
    for j in range(len(id_test)):
        trueclass = d_node_label[id_test[j]][0]
        l_error.append(1.0 - pred_prob[j][trueclass])
    return np.mean(l_error)

def getaccuracy(id_test, pred, pred_prob, d_node_label):
    l_correct = []
    for j in range(len(id_test)):
        trueclass = d_node_label[id_test[j]][0]
        if pred[j] == trueclass:
            l_correct.append(1)
        else:
            l_correct.append(0)
    return np.mean(l_correct)

def readLossofJoel(dataset, f_input):
    d_propPerNodeAvg, d_propPerNodeStd, d_node_label = GetTimeStatisticsPerNode(dataset + "_filtered/preprocessed")
    l_loss = []
    for i in range(10):
        fname = f_input.replace("trial_0", "trial_" + str(i) )
        id_test, pred, pred_prob = readJoels(fname)
        loss = getloss(id_test, pred, pred_prob, d_node_label)
        l_loss.append(loss)
    return np.mean(l_loss), np.std(l_loss)

def readAccuracyofJoel(dataset, f_input):
    d_propPerNodeAvg, d_propPerNodeStd, d_node_label = GetTimeStatisticsPerNode(dataset + "_filtered/preprocessed")
    l_loss = []
    for i in range(10):
        fname = f_input.replace("trial_0", "trial_" + str(i) )
        id_test, pred, pred_prob = readJoels(fname)
        loss = getaccuracy(id_test, pred, pred_prob, d_node_label)
        print loss
        l_loss.append(loss)
    return np.mean(l_loss), np.std(l_loss)

#example usage
if __name__ == "__main__":
    #preprocessed_last_title_attr_filtered
    # generateTrialsbyTGraph("facebook_long_filtered", 10, 0.1)
    # exampleRead()
    # convertToJohn("imdb_binary1_newattr2_selected_60_100")

    # loss, std = readLossofJoel("facebook_long", "/Users/hogun/Downloads/fb_long_and_imdb/facebook_long_filtered/facebook_long_filtered_trial_0_fold_6_PL-EM (CAL).txt")
    # acc, std = readAccuracyofJoel("imdb_binary1_newattr2_long",
    #                            "/Users/hogun/Downloads/fb_long_and_imdb/imdb_binary1_newattr2_long_filtered/imdb_binary1_newattr2_long_filtered_trial_0_fold_6_PL-EM (CAL).txt")
    # acc, std = readAccuracyofJoel("facebook_long",
    #                               "/Users/hogun/Downloads/fb_long_and_imdb/facebook_long_filtered/facebook_long_filtered_trial_0_fold_6_PL-EM (CAL).txt")
    # loss, std = readLossofJoel("facebook_long_all",
    #                            "/Users/hogun/Downloads/fb_imdb_all_long/facebook_long_all_filtered/facebook_long_all_filtered_trial_0_fold_6_PL-EM (CAL).txt")
    print "# acc id:", acc, ", std:", std
    print "# loss:", 1- acc