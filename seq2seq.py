"""
Main script
"""
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

from model.model import EncoderRNN, LuongAttnDecoderRNN
from model.corpus import PAD_token, UNK_token, SOS_token, EOS_token
from util.process_text import prepare_data, replace_UNK, MIN_LENGTH, MAX_LENGTH
from util.object_io import save_object, load_object
from util.text_to_tensor import random_batch
from util.train_helper import train, evaluate_randomly


####################
### Hyper-parameters
####################
USE_CUDA = True
SAVE_OBJECT = False
LOAD_OBJECT = True
# Filter words
MIN_COUNT = 5
VOCAB_SIZE = 10000

####################
### Load files
####################
# Load objects
if LOAD_OBJECT:
    corpus = load_object("./saved/corpus-min-" + str(MIN_COUNT) + "-vocab-" + str(VOCAB_SIZE) + "-lengths-" + str(MIN_LENGTH) + "-" + str(MAX_LENGTH) + ".pkl")
    pairs = load_object("./saved/pairs-min-" + str(MIN_COUNT) + "-vocab-" + str(VOCAB_SIZE) + "-lengths-" + str(MIN_LENGTH) + "-" + str(MAX_LENGTH) + ".pkl")
else:
    # Load file, indexed words and split into pairs
    corpus, pairs = prepare_data("./data/cornell-movie-dialogs.txt")

    corpus.trim(MIN_COUNT)
    corpus.filter_vocab(VOCAB_SIZE)

    # Replace unknown words by UNK token
    pairs = replace_UNK(corpus, pairs)

    # Save objects
    if SAVE_OBJECT:
        save_object(corpus, "./saved/corpus-min-" + str(MIN_COUNT) + "-vocab-" + str(VOCAB_SIZE) + "-lengths-" + str(MIN_LENGTH) + "-" + str(MAX_LENGTH) + ".pkl")
        save_object(pairs, "./saved/pairs-min-" + str(MIN_COUNT) + "-vocab-" + str(VOCAB_SIZE) + "-lengths-" + str(MIN_LENGTH) + "-" + str(MAX_LENGTH) + ".pkl")


##############################
### Train model
##############################
# Model configuration
attn_model = 'dot'
embedding_size = 100
hidden_size = 500
n_layers = 3
dropout = 0.1
batch_size = 128

# Training configuration
clip = 10
teacher_forcing_ratio = 0.5
learning_rate = 0.0001
decoder_learning_ratio = 5
n_epochs = 10000
epoch = 0
print_every = 100
evaluate_every = 500
save_every = 1000

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
plot_losses = []
print_loss_total = 0
plot_loss_total = 0

# Train model
ecs = []
dcs = []
eca = 0
dca = 0

while epoch < n_epochs:
    epoch += 1

    # Get training batch
    input_batches, input_lengths, target_batches, target_lengths = random_batch(corpus, pairs, batch_size)

    # Run the train function
    loss, ec, dc = train(
        input_batches, input_lengths, target_batches, target_lengths,
        encoder, decoder, 
        encoder_optimizer, decoder_optimizer, criterion,
        max_length=MAX_LENGTH, clip=clip
    )

    # Keep track of loss
    print_loss_total += loss
    plot_loss_total += loss
    eca += ec
    dca += dc

    # Print loss
    if epoch % print_every == 0:
        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0
        print_summary = '%s (%d %d%%) %.4f' % (time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
        print(print_summary)

    if epoch % evaluate_every == 0:
        evaluate_randomly(corpus)

    if epoch % save_every == 0:
        torch.save(encoder, "./saved/encoder-min-" + str(MIN_COUNT) + "-vocab-" + str(VOCAB_SIZE) + "-lengths-" + str(MIN_LENGTH) + "-" + str(MAX_LENGTH) + ".pt")
        torch.save(decoder, "./saved/encoder-min-" + str(MIN_COUNT) + "-vocab-" + str(VOCAB_SIZE) + "-lengths-" + str(MIN_LENGTH) + "-" + str(MAX_LENGTH) + ".pt")

    