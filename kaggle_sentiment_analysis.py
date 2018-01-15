import sys
import gc
import time
import math
import random
import pickle
import string
import argparse

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from gensim.models.keyedvectors import KeyedVectors

import nltk
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag

from model.config import WORD2VEC_PATH, AFFECT_EMBEDDING_PATH, AFFECT_EMBEDDING_STRENGTH
from model.model import KaggleSentimentRNN
from util.process_text import normalize_string
from util.text_to_tensor import pad_seq
from util.train_helper import time_since

####################
### Hyper-parameters
####################
parser = argparse.ArgumentParser()

# Global options
globalArgs = parser.add_argument_group('Global options')
globalArgs.add_argument('--GPU_ID', type=int, default=0, help='device id of the GPU used')
globalArgs.add_argument('--model_identifier', type=str, default="ex", help='identifier of this experiment')
globalArgs.add_argument('--SAVE_MODEL', action='store_true', help='If true, save trained model periodically')
globalArgs.add_argument('--LOAD_NUMPY_WORD2VEC', action='store_true', help='If true, load filtered word2vec embedding from numpy array')

# Dataset options
datasetArgs = parser.add_argument_group('Dataset options')
datasetArgs.add_argument('--CORPUS', type=str, default='kaggle_sentiment', help='name of the corpus used')
datasetArgs.add_argument('--KEEP_PUNCTUATIONS', action='store_true', help='If true, keep punctuations in training data')
datasetArgs.add_argument('--MIN_LENGTH', type=int, default=0, help='minimum length of the sentence, define number of minimum step of the RNN')
datasetArgs.add_argument('--MAX_LENGTH', type=int, default=20, help='maximum length of the sentence, define number of maximum step of the RNN')

# Network options
nnArgs = parser.add_argument_group('Network options', 'architecture related option')
nnArgs.add_argument('--rnn_model', type=str, default="gru", help='the type of rnn model used: rnn, gru and lstm')
nnArgs.add_argument('--attention', type=str, default="self-attention", help='the type of attention mechanism used')
nnArgs.add_argument('--hidden_size', type=int, default=512, help='number of hidden units in each RNN cell')
nnArgs.add_argument('--n_layers', type=int, default=1, help='number of rnn layers')
nnArgs.add_argument('--output_projection', action='store_true', help='If true, add another linear projection with nonlinear activation before output layer')
nnArgs.add_argument('--AFFECT_EMBEDDING_STRENGTH', type=float, default=0.01, help='affective word embedding strength')

# Training options
trainingArgs = parser.add_argument_group('Training options')
trainingArgs.add_argument('--n_epochs', type=int, default=1000, help='maximum number of epochs to run')
trainingArgs.add_argument('--batch_size', type=int, default=64, help='mini-batch size')
trainingArgs.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
trainingArgs.add_argument('--dropout', type=float, nargs='+', default=[0, 0.5, 0], help='Dropout rate (keep probabilities)')
trainingArgs.add_argument('--freeze_embed', action='store_true', help='If true, do not tune word embedding')
trainingArgs.add_argument('--LambdaLR', action='store_true', help='If true, decrease learning rate with epochs')
trainingArgs.add_argument('--decrease_lr_every', type=int, default=500, help='Decrease learning rate every a specified number of epochs')
trainingArgs.add_argument('--ReduceLROnPlateau', action='store_true', help='If true, decrease learning rate once validation is not improving')
trainingArgs.add_argument('--decrease_lr_val_factor', type=float, default=0.1, help='The factor of decrease on plateau')
trainingArgs.add_argument('--decrease_lr_val_patience', type=int, default=1, help='The patience if decrease on plateau')
trainingArgs.add_argument('--L2_decay', type=float, default=0.0001, help='L2 regularization coefficient')

args = parser.parse_args(sys.argv[1:])

USE_CUDA = True
GPU_ID = args.GPU_ID
SEED = 412

AFFECT_EMBEDDING_STRENGTH = args.AFFECT_EMBEDDING_STRENGTH
TEST_MODE = False
CORPUS = args.CORPUS

SAVE_MODEL = args.SAVE_MODEL
LOAD_NUMPY_WORD2VEC = args.LOAD_NUMPY_WORD2VEC
MIN_LENGTH = args.MIN_LENGTH
MAX_LENGTH = args.MAX_LENGTH
KEEP_PUNCTUATIONS = args.KEEP_PUNCTUATIONS
val_ratio = 0.2

rnn_model = args.rnn_model
attention = args.attention
embedding_size = 300
hidden_size = args.hidden_size
output_size = 5
n_layers = args.n_layers
output_projection = args.output_projection
dropout = args.dropout

n_epochs = args.n_epochs
epoch = 0
batch_size = args.batch_size
learning_rate = args.learning_rate
freeze_embed = args.freeze_embed
use_LambdaLR = args.LambdaLR
decrease_lr_every = args.decrease_lr_every
use_ReduceLROnPlateau = args.ReduceLROnPlateau
decrease_lr_val_factor = args.decrease_lr_val_factor
decrease_lr_val_patience = args.decrease_lr_val_patience
L2_decay = args.L2_decay
clip = 5
early_stopping = False
model_identifier = args.model_identifier
print_every = 100
save_every = 1000
n_validations = 100

print("Args: ")
print(args)

####################
### Load data
####################

FILE_NAME = "lengths-{0}-{1}-hidden-{2}-layers-{3}-dropout-{4}-epochs-{5}-batch-{6}-learn-{7}-id-{8}".format(
            MIN_LENGTH, MAX_LENGTH, hidden_size, n_layers, dropout, n_epochs,
            batch_size, learning_rate, model_identifier)

# Load sentence data
with open("./data/kaggle_sentiment/train_list.pkl", 'rb') as input:
    train_list = pickle.load(input)

with open("./data/kaggle_sentiment/test_list.pkl", 'rb') as input:
    test_list = pickle.load(input)

inputs = [] # A list of sentences, each sentence has a list of words
targets = [] # A list of VAD tuples, each VAD tuple corresponds to one sentence
tests = []
word2index = {"PAD": 0}
index2word = {0: "PAD"}
word2count = {"PAD": 0}
n_words = 1

# Build corpus dictionaries
print("Building corpus dictionaries...")
start = time.time()
for t in train_list:
    phrase = t[1]
    sent_words = []
    sentiment = t[2] # Use reader perspective

    phrase = normalize_string(phrase.lower())
    words = phrase.split(" ")
    # Remove punctuations
    if not KEEP_PUNCTUATIONS:
        words = [w for w in words if w not in string.punctuation]
    # Filter out very short and very long sentences
    if len(words) >= MIN_LENGTH and len(words) <= MAX_LENGTH:
        for word in words:
            sent_words.append(word)
            if word in word2index:
                word2count[word] += 1
            else:
                word2index[word] = n_words
                word2count[word] = 1
                index2word[n_words] = word
                n_words += 1
        inputs.append(sent_words)
        targets.append(sentiment)

for t in test_list:
    phrase = t[1]
    sent_words = []

    phrase = normalize_string(phrase.lower())
    words = phrase.split(" ")
    # Remove punctuations
    if not KEEP_PUNCTUATIONS:
        words = [w for w in words if w not in string.punctuation]
    # Filter out very short and very long sentences
    for word in words:
        sent_words.append(word)
        if word in word2index:
            word2count[word] += 1
        else:
            word2index[word] = n_words
            word2count[word] = 1
            index2word[n_words] = word
            n_words += 1
    tests.append(sent_words)

print("Indexed {0} words in {1:.2f} seconds".format(n_words, time.time() - start))
print("Selected {0} out of {1} sentences".format(len(inputs), len(train_list)))

sample_index = random.randint(0, len(inputs))
print("Sample input word: ", inputs[sample_index])

# Build training data
for i in range(len(inputs)):
    inputs[i] = [word2index[word] for word in inputs[i]]

print("Sample input word indexes: ", inputs[sample_index])
print("Sample target sentiment value: ", targets[sample_index])

print("Test dataset length: ", len(tests))


####################
### Load embedding
####################

# Word2Vec embeddings
if LOAD_NUMPY_WORD2VEC:
    path = "./saved/{0}/corpus/lengths-{1}-{2}-punc-{3}.npy".format(CORPUS, MIN_LENGTH, MAX_LENGTH, KEEP_PUNCTUATIONS)
    print("Loading word2vec embeddings from " + path)
    embedding = np.load(path)
else:
    print("Loading word2vec embeddings from " + WORD2VEC_PATH)
    start = time.time()
    word_vectors = KeyedVectors.load_word2vec_format(WORD2VEC_PATH, binary=True)
    embedding = np.zeros((n_words, word_vectors.vector_size))
    counter = 0

    for word_id in range(n_words):
        word = index2word[word_id]
        if word in word_vectors.vocab:
            counter += 1
            embedding[word_id] = word_vectors.word_vec(word)
        else:
            embedding[word_id] = np.random.uniform(-0.1, 0.1, size=300)
    
    path = "./saved/{0}/corpus/lengths-{1}-{2}-punc-{3}.npy".format(CORPUS, MIN_LENGTH, MAX_LENGTH, KEEP_PUNCTUATIONS)
    print("Saving word2vec embeddings to " + path)
    np.save(path, embedding)

    print("Finished loading word2vec embeddings in {0:.0f} seconds: {1} out of {2} words are using pre-trained embeddings. ".format(time.time() - start, counter, n_words))
    del word_vectors

# Use affect embedding
affect_embedding = np.zeros((n_words, 3)) # Each word has an affect embedding of size 3
lmtzr = WordNetLemmatizer()
start = time.time()
print("Loading affect embeddings from " + AFFECT_EMBEDDING_PATH)

# Load dictionary of word to vad values 
with open(AFFECT_EMBEDDING_PATH, 'rb') as input:
    word2vad = pickle.load(input)

# Get WordNet POS using treebank POS tag
def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Calculate average vad values
vad_sum = np.array([0.0, 0.0, 0.0])
for word_id in range(n_words):
    word = index2word[word_id]
    if word in string.punctuation:
        continue
    new_word = lmtzr.lemmatize(word, pos=get_wordnet_pos(pos_tag([word])[0][1]))
    if new_word in word2vad:
        vad_sum += np.array(word2vad[new_word])
vad_avg = vad_sum/n_words


# Assign VAD values to each word in corpus
counter = 0
for word_id in range(n_words):
    word = index2word[word_id]
    if word in string.punctuation:
        affect_embedding[word_id] = vad_avg
    else:
        new_word = lmtzr.lemmatize(word, pos=get_wordnet_pos(pos_tag([word])[0][1]))
        
        if new_word in word2vad:
            counter += 1
            affect_embedding[word_id] = np.array(word2vad[new_word])
        else:
            affect_embedding[word_id] = vad_avg
    
print("Finished loading affect embeddings in {0:.0f} seconds: {1} out of {2} words are using pre-trained embeddings. ".format(time.time() - start, counter, n_words))
# Combine word2vec embedding with affect embedding and multiple weights
embedding = np.concatenate((embedding, affect_embedding), axis=1) # (n_words, 300 + output_size)
embedding_size = embedding.shape[1] # Change embedding size to word2vec embedding size
affect_coefficients = np.zeros((embedding_size, 1))
affect_coefficients[: 300] = 1 - AFFECT_EMBEDDING_STRENGTH
affect_coefficients[300: ] = AFFECT_EMBEDDING_STRENGTH
embedding = embedding * affect_coefficients.T # Multiply weights to balance word2vec embedding and affect embedding
del word2vad
gc.collect()


# Random batch
def random_batch(inputs, targets, batch_size):
    input_seqs = []
    target_vad = []

    # Choose random pairs
    for i in range(batch_size):
        idx = random.randint(0, len(inputs) - 1)
        input_seqs.append(inputs[idx])
        target_vad.append(targets[idx])
    
    # Zip into pairs, sort by decreasing length of input sequence, unzip
    seq_pairs = sorted(zip(input_seqs, target_vad), key=lambda p: len(p[0]), reverse=True)
    input_seqs, target_vad = zip(*seq_pairs)

    # For input and output sequences, get array of lengths and pad with PAD_token to max length
    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]

    # Turn padded arrays into (batch_size x max_length) tensors, transpose into (max_length, batch_size)
    input_tensor = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
    target_tensor = Variable(torch.LongTensor(target_vad))

    if USE_CUDA:
        input_tensor = input_tensor.cuda(GPU_ID)
        target_tensor = target_tensor.cuda(GPU_ID)
    
    return input_tensor, input_lengths, target_tensor # (T, B), (B, ), (B, 3)

def accuracy(output, target):
    # output: (batch_size, output_size)
    # target: (batch_size, )
    idx_max = output.max(1)[1]
    accuracy_ = (torch.sum(idx_max == target)).type(torch.FloatTensor)/target.size(0)
    return accuracy_.data[0]

# batch = random_batch(inputs, targets, batch_size)
# print(batch)
# sys.exit()

# Train, validate and test split
print("Spliting into training, validation sets")
random.seed(SEED)

combined = list(zip(inputs, targets))
random.shuffle(combined)
inputs[:], targets[:] = zip(*combined)

val_idx = int(len(inputs) * (1 - val_ratio))
inputs_train, targets_train = inputs[:val_idx], targets[:val_idx]
inputs_val, targets_val = inputs[val_idx:], targets[val_idx:]

####################
### Train model
####################
if TEST_MODE:
    print("Loading model...")
    sentiment_regressor = torch.load("./saved/{0}/{1}/{2}-{3}.{4}".format(CORPUS, "model", "sentiment", FILE_NAME, "pt"))
    
    # Test error
    error_test = 0
    n_test = int(len(inputs_test)/batch_size)
    for i in range(n_test):
        input_tensor, input_lengths, target_tensor = random_batch(inputs_test, targets_test, batch_size)
        output = sentiment_regressor(input_tensor, input_lengths, None)

        # Loss
        loss = criterion(output, target_tensor)
        error_test += loss.data[0]
    
    error_test = error_test/n_test
    print("Test error: ", error_test)
else:
    sentiment_regressor = KaggleSentimentRNN(rnn_model, n_words, embedding_size, hidden_size, output_size, n_layers, dropout, embedding, freeze_embed, output_projection, attention)
    if USE_CUDA:
        sentiment_regressor = sentiment_regressor.cuda(GPU_ID)
    
    # Filter out non-tunable parameters
    parameters = filter(lambda p: p.requires_grad, sentiment_regressor.parameters())
    regressor_optimizer = optim.Adam(parameters, lr=learning_rate, weight_decay=L2_decay)
    lambda_lr = lambda epoch: 0.5 ** (epoch//decrease_lr_every)
    
    scheduler = LambdaLR(regressor_optimizer, lr_lambda=lambda_lr)
    scheduler_on_validation = ReduceLROnPlateau(regressor_optimizer, mode='min', factor=decrease_lr_val_factor, patience=decrease_lr_val_patience, threshold=0.001)
    criterion = nn.CrossEntropyLoss()

    print_loss_total = 0
    print_accuracy_total = 0
    losses_train_all = []
    losses_val_all = []

    while epoch < n_epochs:
        epoch += 1

        if use_LambdaLR:
            scheduler.step()

        # Get batch data
        input_tensor, input_lengths, target_tensor = random_batch(inputs_train, targets_train, batch_size)

        # Clear gradients and loss
        regressor_optimizer.zero_grad()
        loss = 0

        # Forward pass
        output = sentiment_regressor(input_tensor, input_lengths)

        # Loss
        loss = criterion(output, target_tensor)
        batch_accuracy = accuracy(output, target_tensor)

        # Backward
        loss.backward()

        # Gradient clipping
        parameters = filter(lambda p: p.requires_grad, sentiment_regressor.parameters())
        torch.nn.utils.clip_grad_norm(parameters, clip)

        # Optimization
        regressor_optimizer.step()

        loss = loss.data[0]
        print_loss_total += loss
        print_accuracy_total += batch_accuracy
        losses_train_all.append(loss)


        if epoch % print_every == 0:
            # Training error
            print_loss_avg = print_loss_total / print_every
            print_accuracy_avg = print_accuracy_total / print_every
            print_loss_total = 0
            print_accuracy_total = 0

            # Validation error
            error_val = 0
            accuracy_val = 0
            for i in range(n_validations):
                input_tensor, input_lengths, target_tensor = random_batch(inputs_val, targets_val, batch_size)
                output = sentiment_regressor(input_tensor, input_lengths)

                # Loss
                loss = criterion(output, target_tensor)
                batch_accuracy = accuracy(output, target_tensor)
                error_val += loss.data[0]
                accuracy_val += batch_accuracy
            
            error_val = error_val/n_validations
            accuracy_val = accuracy_val/n_validations
            # Adjust learnig rate based on validation error
            if use_ReduceLROnPlateau:
                scheduler_on_validation.step(error_val)

            losses_val_all.append(error_val)

            print_summary = "{0} (Epoch: {1}, Progress: {2:.2f}%) Loss: {3:.4f} Accuracy: {5:.4f}, Validation Loss: {4:.4f}, Validation Accuracy: {6:.4f}".format(
                time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg, error_val, print_accuracy_avg, accuracy_val)
            print(print_summary)

            # Early stopping
            if early_stopping:
                if losses_val_all[-1] > losses_val_all[-2] and losses_val_all[-2] > losses_val_all[-3] and losses_val_all[-3] > losses_val_all[-4]:
                    print("Early stopping: epoch: {0}, training loss: {1:.4f}, validation loss: {2:.4f}".format(epoch, print_loss_avg, error_val))
                    break
            
            # Save losses
            train_loss_file = "./saved/{0}/{1}/{2}-{3}.{4}".format(CORPUS, "loss", "train", FILE_NAME, "txt")
            val_loss_file = "./saved/{0}/{1}/{2}-{3}.{4}".format(CORPUS, "loss", "val", FILE_NAME, "txt")

            with open(train_loss_file,'w') as resultFile:
                for loss in losses_train_all:
                    resultFile.write(str(loss) + '\n')
            
            with open(val_loss_file,'w') as resultFile:
                for loss in losses_val_all:
                    resultFile.write(str(loss) + '\n')


        if epoch % save_every == 0 and SAVE_MODEL:
            torch.save(sentiment_regressor, "./saved/{0}/{1}/{2}-{3}.{4}".format(CORPUS, "model", "sentiment", FILE_NAME, "pt"))

