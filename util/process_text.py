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

# Min and max length for one sentence
MIN_LENGTH = 2
MAX_LENGTH = 80


# Turn a unicode string to plain ASCII
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim and remove non-letter characters
def normalize_string(s):
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
    lines = open(filename).read().strip().split('\n')

    # Split every line into pairs of Q and A
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

    corpus = Corpus(corpus_name)

    return corpus, pairs

# Filter pairs based on min and max lengths, 
def filter_pairs(pairs):
    filter_pairs = []
    for pair in pairs:
        if len(pair[0]) >= MIN_LENGTH and len(pair[0]) <= MAX_LENGTH and len(pair[1]) >= MIN_LENGTH and len(pair[1]) <= MAX_LENGTH:
            filter_pairs.append(pair)
    return filter_pairs

# Filter pairs based on min length, eemove head or tail sentences from a long sentence
def filter_pairs_truncated(pairs):
    filter_pairs = []
    for pair in pairs:
        if len(pair[0]) >= MIN_LENGTH and len(pair[1]) >= MIN_LENGTH:
            if len(pair[0]) <= MAX_LENGTH:
                sents = nltk.sent_tokenize(pair[0])
                new_sent = ''
                # Keep the sentences at the end for questions
                for sent in reversed(sents):
                    if len(new_sent + sent) > MAX_LENGTH:
                        break
                    new_sent = " " + sent + new_sent
                pair[0] = new_sent.strip()
            if len(pair[1]) <= MAX_LENGTH:
                sents = nltk.sent_tokenize(pair[1])
                new_sent = ''
                # Keep the sentences at the begining for answers
                for sent in sents:
                    if len(new_sent + sent) > MAX_LENGTH:
                        break
                    new_sent += sent + " "
                pair[1] = new_sent.strip()
            filter_pairs.append(pair)
    return filter_pairs

# Prepare data using given filename
def prepare_data(filename):
    corpus, pairs = read_corpus(filename)
    print("Read {0} sentence pairs".format(len(pairs)))

    pairs = filter_pairs(pairs)
    print("Filtered to {0} sentence pairs".format(len(pairs)))

    print("Building corpus and indexing words...")
    for pair in pairs:
        corpus.index_words(pair[0])
        corpus.index_words(pair[1])
    print("Indexed {0} words in the corpus".format(corpus.n_words))
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