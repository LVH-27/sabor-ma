#!/usr/bin/python2
# -*- coding: utf-8 -*- 

import cPickle, string, numpy, getopt, sys, random, time, re, pprint
import onlineldavb


batch_size = 64
topic_number = 80

no_of_docs_to_analyze = int(sys.argv[1])
dataset_csv = sys.argv[2]

vocab = file('./cro_vocab.txt').readlines()

olda = onlineldavb.OnlineLDA(vocab,
                             K=30,
                             D=no_of_docs_to_analyze,
                             alpha=1. / topic_number,
                             eta=1. / topic_number,
                             tau0=1024,
                             kappa=0.75
                             )

documents = []
with open(dataset_csv, 'r') as f:
    f.readline()  # read the header to get rid of it
    for i in range(no_of_docs_to_analyze):
        line = f.readline().split(';')
        documents.append(line[1])  # fetch only the transcript

for iteration in range(0, documentstoanalyze):
    gamma, bound = olda.update_lambda_docs(documents)
    wordids, wordcts = onlineldavb.parse_doc_list(documents, olda._vocab)

    perwordbound = bound * len(docset) / (no_of_docs_to_analyze * sum(map(sum, wordcts)))
    print '%d:  rho_t = %f,  held-out perplexity estimate = %f' % \
                (iteration, olda._rhot, numpy.exp(-perwordbound))
    if (iteration % 10 == 0):
        numpy.savetxt('lambda-%d.dat' % iteration, olda._lambda)
        numpy.savetxt('gamma-%d.dat' % iteration, gamma)
