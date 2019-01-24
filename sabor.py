#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import os
import re
from preprocessing import croatian_stemmer as cro_stem
import pickle
import logging
import gensim
from pprint import pprint
from collections import Counter


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
topic_number = 8
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

# with open(vocab_txt, 'r') as f:
#     vocab = f.readlines()

# vocab = [word.strip() for word in vocab]

corpus = get_corpus_csvs(dataset_dir)


if no_of_docs_to_analyze is None:
    no_of_docs_to_analyze = get_corpus_size(corpus)

# ###### CORPUS READING #######

documents = {}  # dictionary of file-granularity list of individual statements
for csv in corpus:
    pickle_path = os.path.join(pickle_dir, os.path.basename(csv))
    try:
        with open(pickle_path, 'rb') as pick:
            print("File {} found - loading CSV {}".format(pickle_path, csv))
            documents[csv] = pickle.load(pick)
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
                    line[i] = cro_stem.stem_document(line[i])
                    doc[header[i]] = line[i]
                else:
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
for doc in documents:
    for entry in documents[doc]:
        # for field in header:
        #     print("{}: {}".format(field, entry[field]))
        # print("")
        transcripts.append(entry["Transkript"])
    break

# ###### PREPROCESSING #######
corpus_file = 'corpus.txt'

freq = Counter()

for transcript in transcripts:
    for token in transcript:
        freq[token] += 1

# drop words occurring only once
transcripts = [[token for token in transcript if freq[token] > 1] for transcript in transcripts]
gensim_dict = gensim.corpora.dictionary.Dictionary(transcripts)

print(len(gensim_dict.token2id))
# gensim_dict.filter_extremes(no_below=0, no_above=0.2, keep_n=None)
gensim_dict.filter_n_most_frequent(int(len(gensim_dict.token2id) * 0.1))
print(len(gensim_dict.token2id))
gensim_dict.save(corpus_file)


gensim_corpus = [gensim_dict.doc2bow(transcript) for transcript in transcripts]
gensim.corpora.MmCorpus.serialize('/tmp/sabor.mm', gensim_corpus)

mm_corpus = gensim.corpora.MmCorpus('/tmp/sabor.mm')

# ###### LDA #######

pickle_path = os.path.join(pickle_dir, 'lda_{}'.format(topic_number))

# if os.path.isfile(pickle_path):
#     print("Loading LDA model from {}...".format(pickle_path))
#     lda = gensim.models.ldamodel.LdaModel.load(pickle_path)

# else:
print("Training the LDA model...")
lda = gensim.models.ldamodel.LdaModel(corpus=mm_corpus,
                                      id2word=gensim_dict,
                                      num_topics=topic_number,
                                      update_every=100,
                                      passes=1
                                      )

pickle_path = os.path.join(pickle_dir, 'lda_{}'.format(topic_number))
print("Saving the LDA model to {}...".format(pickle_dir))
lda.save(pickle_path)

topics = lda.show_topics(topic_number)
with open("topics.out", "w") as f:
    for topic in topics:
        f.write("Topic #{}:\n".format(topic[0]))
        f.write("\t{}\n".format(topic[1]))
# with open("topics.out", "w") as f:
#     f.write('\n'.join(lda.print_topics()))
