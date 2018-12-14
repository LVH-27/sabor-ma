#!/usr/bin/python2

import cPickle, string, numpy, getopt, sys, random, time, re, pprint
import onlineldavb

def main():
	batch_size = 64
	topic_number = 80

	documents_to_analyze = sys.argv[1]
	dataset_csv = sys.argv[2]

	documents = []
	with open(dataset_csv, 'r') as f:
		
