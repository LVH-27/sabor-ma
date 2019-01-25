#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import sys
import os
import re
from preprocessing import croatian_stemmer as cro_stem
import pickle
import logging
import gensim
from pprint import pprint
from collections import Counter


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


# ###### INPUT #######

usage = """
This script performs LDA on Croatian Parliament discussion transcripts. Usage:\n
\tpython {} dataset_csv_directory [number of documents to analyze] [croatian vocabulary file]
""".format(sys.argv[0])
parser = argparse.ArgumentParser(description=usage)
parser.add_argument("-d", "--dataset-dir",
                    help='path to the directory containing the dataset (a set of CSV files)',
                    required=True,
                    type=str)
parser.add_argument("-v", "--vocabulary",
                    help='path to vocabulary file',
                    type=str)
parser.add_argument("-n", "--numdocs",
                    help='total number of documents (CSV entries) to be processed; defaults to whole corpus',
                    type=int)
parser.add_argument("-b", "--batch-size",
                    help='batch size while training the topic model; defaults to 64',
                    type=int)
parser.add_argument("-t", "--topic-number",
                    help='number of topics to detect',
                    required=True,
                    type=int)
parser.add_argument("-p", "--pickle-dir",
                    help='path to directory to which program-specific pickles will be stored; defaults to ./pickles',
                    type=int)

args = parser.parse_args()

corpus = get_corpus_csvs(args.dataset_dir)
if not args.vocabulary:
    args.vocabulary = './cro_vocab.txt'
if not args.batch_size:
    args.batch_size = 64
if not args.pickle_dir:
    args.pickle_dir = 'pickles'
if args.numdocs is None:
    args.numdocs = get_corpus_size(corpus)

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# ###### CORPUS READING #######

documents = {}  # dictionary of file-granularity list of individual statements
for csv in corpus:
    pickle_path = os.path.join(args.pickle_dir, os.path.basename(csv))
    try:
        with open(pickle_path, 'rb') as pick:
            print("File {} found - loading CSV {}".format(pickle_path, csv))
            documents[csv] = pickle.load(pick, encoding='utf-8')
    except FileNotFoundError as e:
        print("File {} not found - reading CSV {}...".format(pickle_path, csv))
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
                    # ###### STEMMING FOR LDA #######
                    stemmed_transcript = cro_stem.stem_document(line[i])
                    # ###### AS-IS FOR AUTHORSHIP ATTRIBUTION #######
                    doc[header[i] + '_stemmed'] = stemmed_transcript
                doc[header[i]] = line[i].strip('\"')
            documents[csv].append(doc)
            if line_no % 100 == 0:
                sys.stdout.write("\r{}".format(line_no))
                sys.stdout.flush()
            line_no += 1

        sys.stdout.write("\r")

        with open(pickle_path, 'wb') as pick:
            pickle.dump(documents[csv], pick)

transcripts = []
transcripts_stemmed = []
for doc in documents:
    for entry in documents[doc]:
        transcripts.append(entry["Transkript"])
        transcripts_stemmed.append(entry["Transkript_stemmed"])


def perform_lda(transcripts, topic_number, batch_size, load_if_existing=False):
    corpus_file = 'corpus.txt'

    freq = Counter()

    for transcript in transcripts:
        for token in transcript:
            freq[token] += 1

    # drop words occurring only once
    transcripts = [[token for token in transcript if freq[token] > 1] for transcript in transcripts]
    gensim_dict = gensim.corpora.dictionary.Dictionary(transcripts)

    gensim_dict.filter_n_most_frequent(int(len(gensim_dict.token2id) * 0.3))
    gensim_dict.save(corpus_file)

    gensim_corpus = [gensim_dict.doc2bow(transcript) for transcript in transcripts]
    gensim.corpora.MmCorpus.serialize('/tmp/sabor.mm', gensim_corpus)

    mm_corpus = gensim.corpora.MmCorpus('/tmp/sabor.mm')

    # ###### LDA #######

    pickle_path = os.path.join(args.pickle_dir, 'lda_t{}_b{}'.format(topic_number, batch_size))

    if os.path.isfile(pickle_path) and load_if_existing:
        print("Loading LDA model from {}...".format(pickle_path))
        lda = gensim.models.ldamodel.LdaModel.load(pickle_path)

    else:
        print("Training the LDA model...")
        lda = gensim.models.ldamodel.LdaModel(corpus=mm_corpus,
                                              id2word=gensim_dict,
                                              num_topics=topic_number,
                                              update_every=batch_size,
                                              passes=3
                                              )

        pickle_path = os.path.join(args.pickle_dir, 'lda_t{}_b{}'.format(topic_number, batch_size))
        print("Saving the LDA model to {}...".format(pickle_path))
        lda.save(pickle_path)

    topics = lda.show_topics(args.topic_number, formatted=False)
    with open("topics.out", "w") as f:
        for topic in topics:
            f.write("Topic #{}:\n".format(topic[0]))
            for word in topic[1]:
                f.write("\t{} - {}\n".format(word[0], word[1]))
