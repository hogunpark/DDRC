from config import config

# Load config parameters
locals().update(config)
import numpy as np
import sys


# helper function to return T or F for True or False. This helps to cut down on fileName size
def bStr(boolStr):
    if boolStr:
        return "T"
    else:
        return "F"


def readConfigurations(dataName="facebook"):
    origName = "data_" + dataName + "_pool_0_pType_none_NNT_CNN_randInit_1_neighbSum_1_pNeighb_0_iterEpoch_1_randop_0_convs_1_convn_1_mpool_4_trial_0"

    # 0    dnType = "RNN"
    # 1    dnType = "CNN"
    # 2    dnType = "CollRNN"
    # 3    dnType = "AttentionRNN"
    # 4    dnType = "AttentionRNNAttr"

    NNTypes = ["AttentionRNNAttr"]
    rInits = [1]
    neighbAvgs = [1]
    pNeighbs = [0]
    iterEpochs = [1]
    poolings = [1]
    randOps = [0]
    mpoolSizes = [4]
    convNums = [1]
    convSizes = [1]
    trials = range(0, 10)
    configurations = []
    count = 0
    for nn in NNTypes:
        for rinit in rInits:
            for mpool in mpoolSizes:
                for convn in convNums:
                    for convs in convSizes:
                        for randop in randOps:
                            for neighbavg in neighbAvgs:
                                for pneighb in pNeighbs:
                                    for iterepoch in iterEpochs:
                                        for pool in poolings:
                                            count += 1
                                            accInput = []
                                            baeInput = []
                                            for i in trials:
                                                newName = origName.replace("pool_0", "pool_" + str(pool)).replace(
                                                    "NNT_CNN", "NNT_" + str(nn)).replace("mpool_0",
                                                                                         "mpool_" + str(mpool)).replace(
                                                    "convn_0", "convn_" + str(convn)).replace("convs_0", "convs_" + str(
                                                    convs)).replace("randop_0", "randop_" + str(randop)).replace(
                                                    "NNT_CollRNN", "NNT_" + nn).replace("randInit_1", "randInit_" + str(
                                                    rinit)).replace("neighbSum_1",
                                                                    "neighbSum_" + str(neighbavg)).replace("pNeighb_0",
                                                                                                           "pNeighb_" + str(
                                                                                                               pneighb)).replace(
                                                    "iterEpoch_1", "iterEpoch_" + str(iterepoch)).replace("trial_0",
                                                                                                          "trial_" + str(
                                                                                                              i))
                                                try:
                                                    x = np.load(outputFolder + newName + ".npy")
                                                    accInput.append(x[0])
                                                    baeInput.append(x[1])
                                                    # accInput.append(count)
                                                    # print(x)
                                                except:
                                                    print(outputFolder + newName + ".npy")
                                            # print(accInput)
                                            xAvgACC = np.mean(np.array(accInput))
                                            xStdACC = np.std(np.array(accInput))
                                            xAvgBAE = np.mean(np.array(baeInput))
                                            xStdBAE = np.std(np.array(baeInput))

                                            tup = (
                                            nn, rinit, neighbavg, pneighb, iterepoch, pool, mpool, convn, convs, randop,
                                            xAvgACC, xStdACC, xAvgBAE, xStdBAE)
                                            configurations.append(tup)

    results = sorted(configurations, key=lambda config: config[-4], reverse=True)
    for tup in results:
        print(
        "NN=" + str(tup[0]) + ", rinit=" + bStr(tup[1]) + ", neighbAvg=" + bStr(tup[2]) + ", perfectNeighbors=" + bStr(
            tup[3]) + ", iterEpoch=" + bStr(tup[4]) + ", pool=" + bStr(tup[5]) + \
        ", mpool=" + str(tup[6]) + ", convn=" + str(tup[7]) + ", convs=" + str(tup[8]) + ", randop=" + str(
            tup[9]) + ", Acc=" + str(tup[10]) + ", Std=" + str(tup[11]) + \
        ", BAE=" + str(tup[12]) + ", Std=" + str(tup[13]) + ", Summary=" + str(tup[10]) + "\t" + str(tup[11]) + \
        "\t" + str(tup[12]) + "\t" + str(tup[13]))


if __name__ == '__main__':
    readConfigurations(dataName=sys.argv[1])
