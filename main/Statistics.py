import networkx as nx
import sys
from collections import defaultdict
from operator import itemgetter
import random
import numpy as np
from operator import add
import theano
from random import shuffle
import copy
import math
from collections import OrderedDict
import os
from config import config
import pickle
# Load config parameters
locals().update(config)


def readHogun(dataFolder, fName):
    f_t_graph  = dataFolder+fName+".p"
    # f_t_graph = "/Users/hogun/Documents/data/tgraph/imdb/preprocessed_binary_filtered_3years_new_attr.p"
    x = pickle.load( open( f_t_graph, "rb" ) )
    nodes = x[0] ; neighborTimeSeries = x[1] ; labels = x[2] ; features = x[3] ;
    G = nx.Graph()
    for i in range(0, len(nodes)):
        x = nodes[i]
        G.add_node(x)
        G.node[x]['label'] = [labels[i]]
        G.node[x]['dynamic_label'] = [float(labels[i])]
        if type(features[i]) is np.ndarray:
            G.node[x]['attr'] = features[i].tolist()
        else:
            G.node[x]['attr'] = features[i].toarray().tolist()[0]
        G.node[x]['neighbors'] = neighborTimeSeries[i]

    return G

def TimeStatistics(fName):
    G = readHogun(dataFolder, fName)
    propWithMean = []
    overallProportions=[]
    for node in G.nodes():
        neighbors = defaultdict(lambda:0)
        proportions = {}
        numTimeSteps = len(G.node[node]['neighbors'])
        for timeStep in G.node[node]['neighbors']:
            #count unique neighbors
            uniqueNeigbhors = {}
            for neighbor in timeStep:
                uniqueNeigbhors[neighbor]=1
            #count frequency
            for neighbor in uniqueNeigbhors:
                neighbors[neighbor]+=1

        for neighbor in neighbors:
            proportions[neighbor] = float(neighbors[neighbor])/numTimeSteps
        propPerNodeAvg = np.mean(proportions.values())
        overallProportions+=proportions.values()
        propWithMean.append(propPerNodeAvg)
    print("mean with mean: "+str(np.mean(propWithMean)))
    print("std with mean: "+str(np.std(propWithMean)))
    print("median with mean: ")+str(np.median(propWithMean))

    print("overall mean: "+str(np.mean(overallProportions)))
    print("overall std: "+str(np.std(overallProportions)))
    print("overall median: ")+str(np.median(overallProportions))

def GetTimeStatisticsPerNode(fName):
    G = readHogun(dataFolder, fName)
    d_propPerNodeAvg = {}
    d_propPerNodeStd = {}
    d_node_label = {}
    for node in G.nodes():
        neighbors = defaultdict(lambda:0)
        proportions = {}
        numTimeSteps = len(G.node[node]['neighbors'])
        d_node_label[node] = G.node[node]['label']

        for timeStep in G.node[node]['neighbors']:
            #count unique neighbors
            uniqueNeigbhors = {}
            for neighbor in timeStep:
                uniqueNeigbhors[neighbor]=1
            #count frequency
            for neighbor in uniqueNeigbhors:
                neighbors[neighbor]+=1


        for neighbor in neighbors:
            proportions[neighbor] = float(neighbors[neighbor])/numTimeSteps

        d_propPerNodeAvg[node] = np.mean(proportions.values())
        d_propPerNodeStd[node] = np.std(proportions.values())

    return d_propPerNodeAvg, d_propPerNodeStd, d_node_label

if __name__ == "__main__":
    fName="facebook"
    #readHogun("../../experiments/data/", "preprocessed_facebook.p")
    #TimeStatistics("facebook_filtered/preprocessed")
    TimeStatistics(str(sys.argv[1])+"/preprocessed")