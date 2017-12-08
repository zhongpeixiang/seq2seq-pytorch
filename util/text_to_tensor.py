"""
This is where all methods for processing sentences into tensors reside
sentences2indexes: Convert each word in a sentence into its index
pad_seq: pad a sequence to max_length using PAD token
random_batch: create a random batch of tensors of size (max_length, batch_size)
"""
import random
import torch
from torch.autograd import Variable

from model.corpus import PAD_token, UNK_token, SOS_token, EOS_token
from model.config import USE_CUDA

# Convert each word in a sentence into its index
def sentences2indexes(corpus, sentence):
    return [corpus.word2index[word] for word in sentence.split(" ")] + [EOS_token]

# Pad a sequence to max_length using PAD token
def pad_seq(seq, max_length):
    seq += [PAD_token] * (max_length - len(seq))
    return seq

# Create a random batch for a specified batch_size
def random_batch(corpus, pairs, batch_size):
    input_seqs = []
    target_seqs = []

    # Choose random pairs
    for i in range(batch_size):
        pair = random.choice(pairs)
        input_seqs.append(sentences2indexes(corpus, pair[0]))
        target_seqs.append(sentences2indexes(corpus, pair[1]))
    
    # Zip into pairs, sort by decreasing length of input sequence, unzip
    seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
    input_seqs, target_seqs = zip(*seq_pairs)

    # For input and output sequences, get array of lengths and pad with PAD_token to max length
    input_lengths = [len(s) for s in input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
    target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]

    # Turn padded arrays into (batch_size x max_length) tensors, transpose into (max_length, batch_size)
    input_tensor = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
    target_tensor = Variable(torch.LongTensor(target_padded)).transpose(0, 1)

    if USE_CUDA:
        input_tensor = input_tensor.cuda()
        target_tensor = target_tensor.cuda()
    
    return input_tensor, input_lengths, target_tensor, target_lengths

