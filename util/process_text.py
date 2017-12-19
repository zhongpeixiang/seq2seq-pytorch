"""
This is where all text processing methods reside.
unicode_to_ascii: convert unicode string to ascii string
normalize_string:  lowercase, trim and remove non-letter characters
read_corpus: read corpus text file and split into pairs of Q and A, and normalize
filter_pairs: filter pairs with required min and max lengths
filter_pairs_truncated: filter pairs based on min length, eemove head or tail sentences from a long sentence
prepare_data: read text file and index words
replace_UNK: replace unknown words by UNK token
"""
import os
import unicodedata
import string
import re
import random
import time
import datetime
import math

import nltk
from model.corpus import Corpus
from model.config import MIN_LENGTH, MAX_LENGTH, REVERSE_INPUT, LIMIT_PAIRS, CORPUS

# Turn a unicode string to plain ASCII
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim and remove non-letter characters
def normalize_string(s):
    if CORPUS == "opensub":
        s = unicode_to_ascii(s.strip())
    else:
        s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([,.!?])", r" \1 ", s)
    s = re.sub(r"[^a-zA-Z,.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()

    # tokenize and combine
    # tokens = nltk.word_tokenize(s)
    # s = " ".join(tokens)
    # s = s.strip()
    return s

# Read txt file, split lines into pairs
def read_corpus(filename):
    print("Reading corpus...")
    corpus_name = os.path.split(filename)[1].split('.')[0]

    # Read the file and split into lines
    print("Spliting lines...")
    start = time.time()
    lines = open(filename).read().strip().split('\n')
    lines = lines[:LIMIT_PAIRS]
    print("Spliting lines took {0:.3f} seconds.".format(time.time() - start))

    # Split every line into pairs of Q and A
    print("Creating pairs...")
    start = time.time()
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]
    print("Creating pairs took {0:.3f} seconds.".format(time.time() - start))

    corpus = Corpus(corpus_name)

    return corpus, pairs

# Filter pairs based on min and max lengths, 
def filter_pairs(pairs):
    filter_pairs = []
    for pair in pairs:
        if len(pair[0].split(" ")) >= MIN_LENGTH and len(pair[0].split(" ")) <= MAX_LENGTH and len(pair[1].split(" ")) >= MIN_LENGTH and len(pair[1].split(" ")) <= MAX_LENGTH:
            if REVERSE_INPUT:
                pair[0] = " ".join([w for w in reversed(pair[0].split(" "))]).strip()
            filter_pairs.append(pair)
    return filter_pairs

# Filter pairs based on min length, eemove head or tail sentences from a long sentence
def filter_pairs_truncated(pairs):
    filter_pairs = []
    for pair in pairs:
        if len(pair[0]) >= MIN_LENGTH and len(pair[1]) >= MIN_LENGTH:
            if len(pair[0].split(" ")) <= MAX_LENGTH:
                sents = nltk.sent_tokenize(pair[0])
                new_sent = ''
                # Keep the sentences at the end for questions
                for sent in reversed(sents):
                    if len((new_sent + sent).split(" ")) > MAX_LENGTH:
                        break
                    new_sent = " " + sent + new_sent
                pair[0] = new_sent.strip()
            if len(pair[1].split(" ")) <= MAX_LENGTH:
                sents = nltk.sent_tokenize(pair[1])
                new_sent = ''
                # Keep the sentences at the begining for answers
                for sent in sents:
                    if len((new_sent + sent).split(" ")) > MAX_LENGTH:
                        break
                    new_sent += sent + " "
                pair[1] = new_sent.strip()
            filter_pairs.append(pair)
    return filter_pairs

# Prepare data using given filename
def prepare_data(filename):
    corpus, pairs = read_corpus(filename)
    print("Read {0} sentence pairs".format(len(pairs)))

    start = time.time()
    pairs = filter_pairs(pairs)
    print("Filtered to {0} sentence pairs, took {1:.3f} seconds".format(len(pairs), time.time() - start))
    

    print("Building corpus and indexing words...")
    start = time.time()
    for pair in pairs:
        corpus.index_words(pair[0])
        corpus.index_words(pair[1])
    print("Indexed {0} words in the corpus, took {1:.3f} seconds".format(corpus.n_words, time.time() - start))
    return corpus, pairs

# Replace unknown words by UNK token
def replace_UNK(corpus, pairs):
    new_pairs = []
    UNK_count = 0
    total_count = 0
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True

        new_input_sentence = ''
        new_output_sentence = ''

        for word in input_sentence.split(' '):
            if word not in corpus.word2index:
                word = 'UNK'
                UNK_count += 1
            new_input_sentence += ' ' + word
            total_count += 1

        for word in output_sentence.split(' '):
            if word not in corpus.word2index:
                word = 'UNK'
                UNK_count += 1
            new_output_sentence += ' ' + word
            total_count += 1

        new_pairs.append((new_input_sentence.strip(), new_output_sentence.strip()))

    print("Replaced {0} UNK in {1} words: {2}%".format(UNK_count, total_count, 100*UNK_count/total_count))
    return new_pairs


# Replace unknown words by UNK token
def remove_UNK(corpus, pairs):
    before_remove_count = len(pairs)
    new_pairs = []

    for pair in pairs:
        keep_input = True
        keep_output = True

        for word in pair[0].split(' '):
            if word not in corpus.word2index:
                keep_input = False

        for word in pair[1].split(' '):
            if word not in corpus.word2index:
                keep_output = False

        if keep_input and keep_output:
            new_pairs.append(pair)
    
    after_remove_count = len(new_pairs)
    print("Keep {0} from {1} pairs; Ratio: {2:.2f}%".format(after_remove_count, before_remove_count, 100*after_remove_count/before_remove_count))
    return new_pairs