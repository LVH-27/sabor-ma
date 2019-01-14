#!/usr/bin/python2
# -*- coding: utf-8 -*-

import sys
import os
import re
from preprocessing import croatian_stemmer as cro_stem
import pickle
from onlineldavb import onlineldavb
import numpy
# import logging
# import gensim
# import random
# import time
# import pprint
# import string
# import getopt


usage = """
This script performs LDA on Croatian Parliament discussion transcripts. Usage:\n
\tpython {} dataset_csv_directory [number of documents to analyze] [croatian vocabulary file]
""".format(sys.argv[0])

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def get_corpus_csvs(dataset_dir):
    transcript_csvs = []
    for f in os.listdir(dataset_dir):
        # fetch only transcripts, not discussion descriptions - that will come later
        if re.search("\.csv$", f):
            transcript_csvs.append(os.path.join(dataset_dir, f))
    return transcript_csvs


def get_corpus_size(corpus):
    size = 0
    for csv in corpus:
        with open(csv, 'r') as f:
            size += len(f.readlines()) - 1  # all lines minus the header
    return size


batch_size = 64
topic_number = 5
pickle_dir = 'pickles'

# ###### INPUT #######

if sys.argv[1] in ['-h', '--help']:
    print(usage)
    exit()
else:
    dataset_dir = sys.argv[1]

try:
    no_of_docs_to_analyze = int(sys.argv[2])
except ValueError:
    no_of_docs_to_analyze = None  # analyze the whole dataset
    vocab_txt = sys.argv[2]
except IndexError:
    no_of_docs_to_analyze = None

try:
    vocab_txt = sys.argv[3]
except IndexError:
    vocab_txt = './cro_vocab.txt'

with open(vocab_txt, 'r') as f:
    vocab = f.readlines()

vocab = [word.decode('utf-8').strip() for word in vocab]

corpus = get_corpus_csvs(dataset_dir)


if no_of_docs_to_analyze is None:
    no_of_docs_to_analyze = get_corpus_size(corpus)

# ###### CORPUS READING #######

documents = {}  # dictionary of file-granularity list of individual statements
for csv in corpus:
    print("Reading CSV {}...".format(csv))
    with open(csv, 'r') as input_csv:
        lines = [line.strip() for line in input_csv.readlines()]

    header = [column.strip('\"') for column in lines[0].split(';')]  # get header and strip quotation marks
    lines = lines[1:]  # drop the header

    documents[csv] = []
    line_no = 0
    for line in lines:
        line = line.split(';')
        doc = {}
        for i in range(len(header)):
            if header[i] == "Transkript":
                line[i] = cro_stem.stem_document(line[i])
                doc[header[i]] = line[i]
            else:
                doc[header[i]] = line[i].strip('\"')
        documents[csv].append(doc)
        sys.stdout.write("\r{}".format(line_no))
        sys.stdout.flush()
        line_no += 1

    sys.stdout.write("\r")

    pickle_path = os.path.join(pickle_dir, os.path.basename(csv))

    with open(pickle_path, 'wb') as pick:
        pickle.dump(documents[csv], pick)

transcripts = []
for doc in documents:
    for entry in documents[doc][:10]:
        # for field in header:
        #     print("{}: {}".format(field, entry[field]))
        # print("")
        transcripts.append(' '.join(entry["Transkript"]))
    break

# ###### LDA #######

olda = onlineldavb.OnlineLDA(vocab,
                             K=topic_number,
                             D=no_of_docs_to_analyze,
                             alpha=1. / topic_number,  # uniform Dirichlet prior parameter: per-document topic mixture
                             eta=1. / topic_number,  # uniform Dirichlet prior parameter: per-corpus topic mixture
                             tau0=1024,  # positive learning parameter that downweights early iterations
                             kappa=0.75  # learning rate
                             )

# print(transcripts)
out_dir = 'out'
for iteration in range(0, no_of_docs_to_analyze):
    gamma, bound = olda.update_lambda_docs(transcripts)
    wordids, wordcts = onlineldavb.parse_doc_list(transcripts, olda._vocab)

    perwordbound = bound * len(transcripts) / (no_of_docs_to_analyze * sum(map(sum, wordcts)))
    print('{}:  rho_t = {},  held-out perplexity estimate = {}'.format(iteration, olda._rhot, numpy.exp(-perwordbound)))
    if (iteration % 10 == 0):
        numpy.savetxt(os.path.join(out_dir, 'lambda-%d.dat') % iteration, olda._lambda)
        numpy.savetxt(os.path.join(out_dir, 'gamma-%d.dat') % iteration, gamma)

