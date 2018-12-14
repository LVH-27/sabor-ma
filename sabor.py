#!/usr/bin/python2
# -*- coding: utf-8 -*- 

import cPickle, string, numpy, getopt, sys, random, time, re, pprint
import onlineldavb


batch_size = 64
topic_number = 80

no_of_docs_to_analyze = int(sys.argv[1])
dataset_csv = sys.argv[2]

documents = []
with open(dataset_csv, 'r') as f:
	f.readline()  # read the header to get rid of it
	for i in range(no_of_docs_to_analyze):
		line = f.readline().split(';')
		documents.append(line[1])  # fetch only the transcript
for document in documents:
	print(document)
	
