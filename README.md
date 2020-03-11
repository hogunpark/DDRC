Deep Dynamic Relational Classifiers (DDRC)
===============================

This is a python implementation of Deep Dynamic Relational Classifiers (DDRC) for the task of (semi-supervised) classification of nodes in a graph, as described in our paper:

**Hogun Park**, John Moore, Jennifer Neville, Deep Dynamic Relational Classifiers: Exploiting Dynamic Neighborhoods in Complex Networks, Proc. of MAISoN Workshop in 10th ACM Conference on Web Search and Data Mining (WSDM 2017), 2017.

Usage
-----

**Example Usage**
    ``python arg_rnn_classifier_with_attr.py --dataset dblp --hiddenunits 256 --timedistunits 64 --maxpooling 4 --randoption 0 --dntypeoption 0 --epochs 500 --nclasses 6 --preprocesseddataset 0 --parallel 1``

Requirements
------------
* numpy
* scipy
* keras
* theano

(may have to be independently  installed)


