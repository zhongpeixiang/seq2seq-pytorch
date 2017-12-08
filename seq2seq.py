"""
Main script
"""
import time
import math
from random import shuffle

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


####################
### Load files
####################
# Load objects
if LOAD_OBJECT:
    corpus = load_object("./saved/corpus-min-" + str(MIN_COUNT) + "-vocab-" + str(VOCAB_SIZE) + "-lengths-" + str(MIN_LENGTH) + "-" + str(MAX_LENGTH) + ".pkl")
    pairs = load_object("./saved/pairs-min-" + str(MIN_COUNT) + "-vocab-" + str(VOCAB_SIZE) + "-lengths-" + str(MIN_LENGTH) + "-" + str(MAX_LENGTH) + ".pkl")
else:
    # Load file, indexed words and split into pairs
    corpus, pairs = prepare_data("./data/cornell-movie-dialogs.txt", MIN_LENGTH, MAX_LENGTH)

    corpus.trim(MIN_COUNT)
    corpus.filter_vocab(VOCAB_SIZE)

    # Replace unknown words by UNK token
    pairs = replace_UNK(corpus, pairs)

    # Save objects
    if SAVE_OBJECT:
        save_object(corpus, "./saved/corpus-min-{0}-vocab-{1}-lengths-{2}-{3}.pkl".format(MIN_COUNT, VOCAB_SIZE, MIN_LENGTH, MAX_LENGTH))
        save_object(pairs, "./saved/pairs-min-{0}-vocab-{1}-lengths-{2}-{3}.pkl".format(MIN_COUNT, VOCAB_SIZE, MIN_LENGTH, MAX_LENGTH))
        
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

# Initialize model
encoder = EncoderRNN(corpus.n_words, embedding_size, hidden_size, n_layers, dropout=dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding_size, hidden_size, corpus.n_words, n_layers, dropout=dropout)

# Initialize optimizers and criterion
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

if USE_CUDA:
    encoder.cuda()
    decoder.cuda()

start = time.time()
print_loss_total = 0

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

    # Print loss
    if epoch % print_every == 0:
        # Training error
        print_loss_avg = print_loss_total / print_every
        perplexity = math.exp(float(print_loss_avg)) if print_loss_avg < 300 else float("inf")
        print_loss_total = 0

        # Validation error
        input_batches, input_lengths, target_batches, target_lengths = random_batch(corpus, pairs_val, batch_size)
        error_val = validate(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder)
        perplexity_val = math.exp(float(error_val)) if error_val < 300 else float("inf")

        print_summary = "{0} (Epoch: {1}, Progress {2}) Loss: {3}, Perplexity: {4}. Validation Loss: {5}, Validation Perplexity: {6}".format(
            time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg, perplexity, error_val, perplexity_val)
        print(print_summary)

    if epoch % evaluate_every == 0:
        # Evaluate random samples from test set
        evaluate_randomly(corpus, pairs_test, encoder, decoder, MAX_LENGTH)

    if epoch % save_every == 0:
        torch.save(encoder, "./saved/encoder-min-{0}-vocab-{1}-lengths-{2}-{3}-atten-{4}-embed-{5}-hidden-{6}-layers-{7}-dropout-{8}-batch-{9}.pt".format(
            MIN_COUNT, VOCAB_SIZE, MIN_LENGTH, MAX_LENGTH, attn_model, embedding_size, hidden_size, n_layers, dropout, batch_size))
        torch.save(decoder, "./saved/decoder-min-{0}-vocab-{1}-lengths-{2}-{3}-atten-{4}-embed-{5}-hidden-{6}-layers-{7}-dropout-{8}-batch-{9}.pt".format(
            MIN_COUNT, VOCAB_SIZE, MIN_LENGTH, MAX_LENGTH, attn_model, embedding_size, hidden_size, n_layers, dropout, batch_size))

    