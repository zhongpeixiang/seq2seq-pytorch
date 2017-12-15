"""
Main script
"""
import time
import math
from random import shuffle

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

from model.model import EncoderRNN, LuongAttnDecoderRNN
from model.corpus import PAD_token, UNK_token, SOS_token, EOS_token
from util.process_text import prepare_data, replace_UNK
from util.object_io import save_object, load_object
from util.text_to_tensor import random_batch
from util.train_helper import train, validate, evaluate_randomly, time_since

# Load config file
from model.config import *

# Print config file
with open("./model/config.py", 'r') as fin:
    print(fin.read())


####################
### Load files
####################
# Load objects
if LOAD_CORPUS:
    print("Loading model...")
    corpus = load_object("./saved/{5}/corpus/corpus-min-{0}-vocab-{1}-lengths-{2}-{3}-reversed-{4}.pkl".format(MIN_COUNT, VOCAB_SIZE, MIN_LENGTH, MAX_LENGTH, REVERSE_INPUT, CORPUS))
    pairs = load_object("./saved/{5}/corpus/pairs-min-{0}-vocab-{1}-lengths-{2}-{3}-reversed-{4}.pkl".format(MIN_COUNT, VOCAB_SIZE, MIN_LENGTH, MAX_LENGTH, REVERSE_INPUT, CORPUS))
else:
    # Load file, indexed words and split into pairs
    corpus, pairs = prepare_data("./data/cornell-movie-dialogs.txt")

    corpus.trim(MIN_COUNT)
    corpus.filter_vocab(VOCAB_SIZE)

    # Replace unknown words by UNK token
    pairs = replace_UNK(corpus, pairs)

    # Save objects
    if SAVE_CORPUS:
        save_object(corpus, "./saved/{5}/corpus/corpus-min-{0}-vocab-{1}-lengths-{2}-{3}-reversed-{4}.pkl".format(MIN_COUNT, VOCAB_SIZE, MIN_LENGTH, MAX_LENGTH, REVERSE_INPUT, CORPUS))
        save_object(pairs, "./saved/{5}/corpus/pairs-min-{0}-vocab-{1}-lengths-{2}-{3}-reversed-{4}.pkl".format(MIN_COUNT, VOCAB_SIZE, MIN_LENGTH, MAX_LENGTH, REVERSE_INPUT, CORPUS))
        
# Train, validation and test split
print("Spliting into training, validation and test sets")
shuffle(pairs)
val_idx = int(len(pairs) * (1 - val_ratio - test_ratio))
test_idx = int(len(pairs) * (1 - test_ratio))
pairs_train = pairs[:val_idx]
pairs_val = pairs[val_idx:test_idx]
pairs_test = pairs[test_idx:]


##############################
### Train model
##############################
if LOAD_MODEL:
    encoder = torch.load("./saved/{12}/model/encoder-min-{0}-vocab-{1}-lengths-{2}-{3}-atten-{4}-embed-{5}-hidden-{6}-layers-{7}-dropout-{8}-batch-{9}-teacher-{10}-learn-{11}.pt".format(
            MIN_COUNT, VOCAB_SIZE, MIN_LENGTH, MAX_LENGTH, attn_model, embedding_size, hidden_size, n_layers, dropout, batch_size, teacher_forcing_ratio, learning_rate, CORPUS))
    decoder = torch.load("./saved/{12}/model/decoder-min-{0}-vocab-{1}-lengths-{2}-{3}-atten-{4}-embed-{5}-hidden-{6}-layers-{7}-dropout-{8}-batch-{9}-teacher-{10}-learn-{11}.pt".format(
            MIN_COUNT, VOCAB_SIZE, MIN_LENGTH, MAX_LENGTH, attn_model, embedding_size, hidden_size, n_layers, dropout, batch_size, teacher_forcing_ratio, learning_rate, CORPUS))

    print("Start evaluating...")
    while epoch < n_epochs:
        epoch += 1
        evaluate_randomly(corpus, pairs_test, encoder, decoder, 10)
else:
    # Initialize model
    encoder = EncoderRNN(corpus.n_words, embedding_size, hidden_size, n_layers, dropout=dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding_size, hidden_size, corpus.n_words, n_layers, dropout=dropout)

    # Initialize optimizers and criterion
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    criterion = nn.CrossEntropyLoss()

    if USE_CUDA:
        encoder.cuda(GPU_ID)
        decoder.cuda(GPU_ID)

    start = time.time()
    print_loss_total = 0
    losses_train_all = []
    losses_train = []
    losses_val_all = []
    losses_val = []

    print("Start training...")
    while epoch < n_epochs:
        epoch += 1

        # Get training batch
        input_batches, input_lengths, target_batches, target_lengths = random_batch(corpus, pairs_train, batch_size)

        # Run the train function
        loss, ec, dc = train(
            input_batches, input_lengths, target_batches, target_lengths,
            encoder, decoder, 
            encoder_optimizer, decoder_optimizer, criterion,
            clip=clip
        )

        # Keep track of loss
        print_loss_total += loss
        losses_train_all.append(loss)


        # Print loss
        if epoch % print_every == 0:
            # Training error
            print_loss_avg = print_loss_total / print_every
            perplexity = math.exp(float(print_loss_avg)) if print_loss_avg < 300 else float("inf")
            print_loss_total = 0
            losses_train.append(print_loss_avg)

            # Validation error
            input_batches, input_lengths, target_batches, target_lengths = random_batch(corpus, pairs_val, batch_size)
            error_val = validate(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder)
            perplexity_val = math.exp(float(error_val)) if error_val < 300 else float("inf")
            losses_val.append(error_val)
            losses_val_all.append(error_val)

            print_summary = "{0} (Epoch: {1}, Progress: {2:.2f}%) Loss: {3:.2f}, Perplexity: {4:.2f}. Validation Loss: {5:.2f}, Validation Perplexity: {6:.2f}".format(
                time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg, perplexity, error_val, perplexity_val)
            print(print_summary)

            # Early stopping
            if early_stopping:
                if losses_val[-1] > losses_val[-2] and losses_val[-2] > losses_val[-3] and losses_val[-3] > losses_val[-4]:
                    print("Early stopping: epoch: {0}, training loss: {1:.2f}, validation loss: {2:.2f}".format(epoch, print_loss_avg, error_val))
                    break
            
            # Save losses
            train_loss_file = "./saved/{12}/loss/train-loss-min-{0}-vocab-{1}-lengths-{2}-{3}-atten-{4}-embed-{5}-hidden-{6}-layers-{7}-dropout-{8}-batch-{9}-teacher-{10}-learn-{11}.txt".format(
                MIN_COUNT, VOCAB_SIZE, MIN_LENGTH, MAX_LENGTH, attn_model, embedding_size, hidden_size, n_layers, dropout, batch_size, teacher_forcing_ratio, learning_rate, CORPUS)
            val_loss_file = "./saved/{12}/loss/val-loss-min-{0}-vocab-{1}-lengths-{2}-{3}-atten-{4}-embed-{5}-hidden-{6}-layers-{7}-dropout-{8}-batch-{9}-teacher-{10}-learn-{11}.txt".format(
                MIN_COUNT, VOCAB_SIZE, MIN_LENGTH, MAX_LENGTH, attn_model, embedding_size, hidden_size, n_layers, dropout, batch_size, teacher_forcing_ratio, learning_rate, CORPUS)
            
            with open(train_loss_file,'w') as resultFile:
                for loss in losses_train_all:
                    resultFile.write(str(loss) + '\n')
            
            with open(val_loss_file,'w') as resultFile:
                for loss in losses_val_all:
                    resultFile.write(str(loss) + '\n')

        if epoch % evaluate_every == 0:
            # Evaluate random samples from test set
            evaluate_randomly(corpus, pairs_test, encoder, decoder, 10)

        if epoch % save_every == 0 and SAVE_MODEL:
            torch.save(encoder, "./saved/{12}/model/encoder-min-{0}-vocab-{1}-lengths-{2}-{3}-atten-{4}-embed-{5}-hidden-{6}-layers-{7}-dropout-{8}-batch-{9}-teacher-{10}-learn-{11}.pt".format(
                MIN_COUNT, VOCAB_SIZE, MIN_LENGTH, MAX_LENGTH, attn_model, embedding_size, hidden_size, n_layers, dropout, batch_size, teacher_forcing_ratio, learning_rate, CORPUS))
            torch.save(decoder, "./saved/{12}/model/decoder-min-{0}-vocab-{1}-lengths-{2}-{3}-atten-{4}-embed-{5}-hidden-{6}-layers-{7}-dropout-{8}-batch-{9}-teacher-{10}-learn-{11}.pt".format(
                MIN_COUNT, VOCAB_SIZE, MIN_LENGTH, MAX_LENGTH, attn_model, embedding_size, hidden_size, n_layers, dropout, batch_size, teacher_forcing_ratio, learning_rate, CORPUS))

        