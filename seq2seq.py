"""
Main script
"""
import sys
import gc
import time
import math
import random
import pickle

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

from model.model import EncoderRNN, LuongAttnDecoderRNN
from model.corpus import PAD_token, UNK_token, SOS_token, EOS_token
from util.process_text import prepare_data, replace_UNK, remove_UNK
from util.object_io import save_object, load_object
from util.text_to_tensor import random_batch, get_ordered_batch
from util.train_helper import train, validate, evaluate_randomly, time_since

# Load config file
from model.config import *

# Print config file
with open("./model/config.py", 'r') as fin:
    print(fin.read())

print()
print("####################")
print("### Corpus Details")
print("####################")
print()
####################
### Load files
####################
# Load objects
if LOAD_CORPUS:
    corpus = load_object("./saved/{5}/corpus/corpus-min-{0}-vocab-{1}-lengths-{2}-{3}-replace-{6}-reversed-{4}.pkl".format(MIN_COUNT, VOCAB_SIZE, MIN_LENGTH, MAX_LENGTH, REVERSE_INPUT, CORPUS, REPLACE_UNK))
    pairs = load_object("./saved/{5}/corpus/pairs-min-{0}-vocab-{1}-lengths-{2}-{3}-replace-{6}-reversed-{4}.pkl".format(MIN_COUNT, VOCAB_SIZE, MIN_LENGTH, MAX_LENGTH, REVERSE_INPUT, CORPUS, REPLACE_UNK))
else:
    # Load file, indexed words and split into pairs
    corpus, pairs = prepare_data(USE_DIR + DATA_FILE)

    # Filter words
    print("Triming words...")
    start = time.time()
    corpus.trim(MIN_COUNT)
    print("Triming took {0:.3f} seconds".format(time.time() - start))

    print("Limiting vocab size...")
    start = time.time()
    corpus.filter_vocab(VOCAB_SIZE)
    print("Limiting vocab size took {0:.3f} seconds".format(time.time() - start))

    if REPLACE_UNK:
        # Replace unknown words by UNK token
        pairs = replace_UNK(corpus, pairs)
    else:
        # Remove pairs that contain unknown words
        pairs = remove_UNK(corpus, pairs)

    # Save objects
    if SAVE_CORPUS:
        save_object(corpus, "./saved/{5}/corpus/corpus-min-{0}-vocab-{1}-lengths-{2}-{3}-replace-{6}-reversed-{4}.pkl".format(MIN_COUNT, VOCAB_SIZE, MIN_LENGTH, MAX_LENGTH, REVERSE_INPUT, CORPUS, REPLACE_UNK))
        save_object(pairs, "./saved/{5}/corpus/pairs-min-{0}-vocab-{1}-lengths-{2}-{3}-replace-{6}-reversed-{4}.pkl".format(MIN_COUNT, VOCAB_SIZE, MIN_LENGTH, MAX_LENGTH, REVERSE_INPUT, CORPUS, REPLACE_UNK))
    
    # If intend to create corpus only, exit now
    if CREATE_CORPUS_ONLY:
        sys.exit()

# Print corpus info
print("Vocab size: ", corpus.n_words)
print("Number of samples: ", len(pairs))
num_words = 0
for pair in pairs:
    num_words += len(pairs[0])
    num_words += len(pairs[1])
print("Number of words: ", num_words)
print("Sample pair: ")
print(pairs[random.randint(0, len(pairs) - 1)])


####################
### Load embedding
####################
def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Load word2vec embedding
embedding = None
if LOAD_WORD2VEC:
    print("Initializing word2vec embeddings...")
    if LOAD_NUMPY_WORD2VEC:
        path = "./saved/{5}/corpus/corpus-min-{0}-vocab-{1}-lengths-{2}-{3}-replace-{6}-reversed-{4}.npy".format(MIN_COUNT, VOCAB_SIZE, MIN_LENGTH, MAX_LENGTH, REVERSE_INPUT, CORPUS, REPLACE_UNK)
        print("Loading word2vec embeddings from " + path)
        embedding = np.load(path)
    else:
        print("Loading word2vec embeddings from " + WORD2VEC_PATH)
        start = time.time()
        word_vectors = KeyedVectors.load_word2vec_format(WORD2VEC_PATH, binary=True)
        embedding = np.zeros((corpus.n_words, word_vectors.vector_size))
        counter = 0

        for word_id in range(corpus.n_words):
            word = corpus.index2word[word_id]
            if word in word_vectors.vocab:
                counter += 1
                embedding[word_id] = word_vectors.word_vec(word)
            else:
                embedding[word_id] = np.random.normal(loc=0, scale=1/np.sqrt(word_vectors.vector_size), size=word_vectors.vector_size)
        
        if SAVE_NUMPY_WORD2VEC:
            path = "./saved/{5}/corpus/corpus-min-{0}-vocab-{1}-lengths-{2}-{3}-replace-{6}-reversed-{4}.npy".format(MIN_COUNT, VOCAB_SIZE, MIN_LENGTH, MAX_LENGTH, REVERSE_INPUT, CORPUS, REPLACE_UNK)
            print("Saving word2vec embeddings to " + path)
            np.save(path, embedding)
        
        print("Finished loading word2vec embeddings in {0:.0f} seconds: {1} out of {2} words are using pre-trained embeddings. ".format(time.time() - start, counter, corpus.n_words))
        del word_vectors

    # Keep a copy of original word2vec embedding
    word_embedding = nn.Embedding(corpus.n_words, 300)
    word_embedding.weight.data.copy_(torch.from_numpy(embedding))
    word_embedding = word_embedding.cuda(GPU_ID)

    # Load affect embedding dictionary
    if USE_AFFECT_EMBEDDING:
        affect_embedding = np.zeros((corpus.n_words, 3)) # Each word has an affect embedding of size 3
        lmtzr = WordNetLemmatizer()
        start = time.time()
        print("Loading affect embeddings from " + AFFECT_EMBEDDING_PATH)

        # Load dictionary of word to vad values 
        with open(AFFECT_EMBEDDING_PATH, 'rb') as input:
            word2vad = pickle.load(input)

        # Calculate average vad values
        vad_sum = np.array([0.0, 0.0, 0.0])
        for word_id in range(corpus.n_words):
            word = corpus.index2word[word_id]
            new_word = lmtzr.lemmatize(word, pos=get_wordnet_pos(pos_tag([word])[0][1]))
            if new_word in word2vad:
                vad_sum += np.array(word2vad[new_word])
        vad_avg = vad_sum/corpus.n_words
        
        lemma_corpus_vocab = set()
        lemma_vad_vocab = set()

        # Assign VAD values to each word in corpus
        for word_id in range(corpus.n_words):
            word = corpus.index2word[word_id]
            new_word = lmtzr.lemmatize(word, pos=get_wordnet_pos(pos_tag([word])[0][1]))
            lemma_corpus_vocab.add(new_word)
            
            if new_word in word2vad:
                lemma_vad_vocab.add(new_word)
                affect_embedding[word_id] = np.array(word2vad[new_word])
            else:
                affect_embedding[word_id] = vad_avg
            
        print("Finished loading affect embeddings in {0:.0f} seconds: {1} out of {2} words are using pre-trained embeddings. ".format(time.time() - start, len(lemma_vad_vocab), len(lemma_corpus_vocab)))
        # Combine word2vec embedding with affect embedding and multiple weights
        embedding = np.concatenate((embedding, affect_embedding), axis=1) # (n_words, 303)
        embedding_size = embedding.shape[1] # Change embedding size to word2vec embedding size
        affect_coefficients = np.zeros((embedding_size, 1))
        affect_coefficients[: 300] = 1 - AFFECT_EMBEDDING_STRENGTH
        affect_coefficients[300: ] = AFFECT_EMBEDDING_STRENGTH
        embedding = embedding * affect_coefficients.T # Multiply weights to balance word2vec embedding and affect embedding
        del word2vad
        
        # Keep a copy of original affect embedding
        affect_embedding_copy = nn.Embedding(corpus.n_words, 3)
        affect_embedding_copy.weight.data.copy_(torch.from_numpy(affect_embedding))
        affect_embedding_copy = affect_embedding_copy.cuda(GPU_ID)

# Train, validation and test split
print("Spliting into training, validation and test sets")
random.seed(SEED)
random.shuffle(pairs)
val_idx = int(len(pairs) * (1 - val_ratio - test_ratio))
test_idx = int(len(pairs) * (1 - test_ratio))
pairs_train = pairs[:val_idx]
pairs_val = pairs[val_idx:test_idx]
pairs_test = pairs[test_idx:]

# Sort pairs by input sequence length to make samples in a batch have equal lengths
if ordered_batch:
    print("Sorting training samples by input sequence length in ascending order...")
    pairs_train = sorted(pairs_train, key=lambda pair: len(pair[0].split(" ")))
    current_index = 0

del pairs
gc.collect()
##############################
### Train model
##############################
print()
print("####################")
print("### Training Details")
print("####################")
print()

FILE_NAME = "min-{0}-vocab-{1}-lengths-{2}-{3}-affect-{15}-atten-{4}-embed-{5}-word2vec-{12}-hidden-{6}-layers-{7}-dropout-{8}-epochs-{14}-batch-{9}-teacher-{10}-learn-{11}-id-{13}".format(
            MIN_COUNT, VOCAB_SIZE, MIN_LENGTH, MAX_LENGTH, attn_model, embedding_size, hidden_size, n_layers, dropout, 
            str(ordered_batch) + "_" + str(batch_size), teacher_forcing_ratio, learning_rate, LOAD_WORD2VEC, model_identifier, n_epochs, 
            str(USE_AFFECT_EMBEDDING) + "_" + str(AFFECT_EMBEDDING_STRENGTH) + "_" + str(AFFECT_ATTN) + "_" + str(AFFECT_LOSS_STRENGTH))

if LOAD_MODEL:
    print("Loading model from ", "./saved/{0}/{1}/{2}-{3}.{4}".format(CORPUS, "model", "encoder", FILE_NAME, "pt"))
    encoder = EncoderRNN(corpus.n_words, embedding_size, hidden_size, n_layers, dropout=dropout, embedding=embedding)
    decoder = LuongAttnDecoderRNN(attn_model, bigram_attn, embedding_size, hidden_size, corpus.n_words, n_layers, dropout=dropout, embedding=embedding, affect_attn=AFFECT_ATTN)
    encoder.load_state_dict(torch.load("./saved/{0}/{1}/{2}-{3}.{4}".format(CORPUS, "model", "encoder", FILE_NAME, "pt")))
    decoder.load_state_dict(torch.load("./saved/{0}/{1}/{2}-{3}.{4}".format(CORPUS, "model", "decoder", FILE_NAME, "pt")))

    if USE_CUDA:
        encoder.cuda(GPU_ID)
        decoder.cuda(GPU_ID)
    
    print("Start evaluating...")
    while epoch < n_epochs:
        epoch += 1
        evaluate_randomly(corpus, pairs_test, encoder, decoder, MAX_LENGTH)
else:
    # Initialize model
    encoder = EncoderRNN(corpus.n_words, embedding_size, hidden_size, n_layers, dropout=dropout, embedding=embedding)
    decoder = LuongAttnDecoderRNN(attn_model, bigram_attn, embedding_size, hidden_size, corpus.n_words, n_layers, dropout=dropout, embedding=embedding, affect_attn=AFFECT_ATTN)
    
    # Total number of params
    encoder_num_params = 0
    decoder_num_params = 0
    for param in encoder.parameters():
        product = 1
        for dim in range(len(param.data.size())):
            product *= param.data.size(dim)
        encoder_num_params += product
    
    for param in decoder.parameters():
        product = 1
        for dim in range(len(param.data.size())):
            product *= param.data.size(dim)
        decoder_num_params += product
    # encoder_num_params = 6*hidden_size*((3*n_layers - 2)*hidden_size + embedding_size + 2*n_layers)
    # decoder_num_params = hidden_size*((6*n_layers - 1)*hidden_size + 3*embedding_size + 6*n_layers + corpus.n_words + 1) + corpus.n_words
    print("Number of params: encoder ({0:,}), decoder ({1:,}), total ({2:,})".format(encoder_num_params, decoder_num_params, encoder_num_params + decoder_num_params))

    # Initialize optimizers and criterion
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=L2_decay)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio, weight_decay=L2_decay)
    criterion = nn.CrossEntropyLoss()

    # Learning rate annealing
    lambda_lr = lambda epoch: 0.5 ** (epoch//decrease_lr_every)
    encoder_scheduler = LambdaLR(encoder_optimizer, lr_lambda=lambda_lr)
    encoder_scheduler_on_validation = ReduceLROnPlateau(encoder_optimizer, mode='min', factor=decrease_lr_val_factor, patience=decrease_lr_val_patience, threshold=0.001)
    decoder_scheduler = LambdaLR(decoder_optimizer, lr_lambda=lambda_lr)
    decoder_scheduler_on_validation = ReduceLROnPlateau(decoder_optimizer, mode='min', factor=decrease_lr_val_factor, patience=decrease_lr_val_patience, threshold=0.001)

    if USE_CUDA:
        encoder.cuda(GPU_ID)
        decoder.cuda(GPU_ID)

    start = time.time()
    print_loss_total = 0
    losses_train_all = []
    losses_train = []
    losses_val_all = []
    losses_val = []

    # Create a file to save validation metrics
    val_metrics_file = "./saved/{0}/{1}/{2}-{3}.{4}".format(CORPUS, "loss", "metrics", FILE_NAME, "txt")
    with open(val_metrics_file, "w") as f:
        f.write("loss,perplexity,distinct-1,distinct-2,embed_greedy,embed_avg,embed_extrema,affect_distance,affect_strength\n")

    print("Start training...")
    while epoch < n_epochs:
        epoch += 1

        if use_LambdaLR:
            encoder_scheduler.step()
            decoder_scheduler.step()
        
        # Learning rate annealing
        # if epoch % 10000 == 0 or epoch % 15000 == 0 or epoch % 18000 == 0:
        #     learning_rate = learning_rate/2

        # Get training batch
        # print("Creating a random batch ...")
        # start = time.time()
        if ordered_batch:
            input_batches, input_lengths, target_batches, target_lengths = get_ordered_batch(corpus, pairs_train, batch_size, current_index)
            current_index += batch_size
        else:
            input_batches, input_lengths, target_batches, target_lengths = random_batch(corpus, pairs_train, batch_size)
        # print("Creating a random batch took {0:.3f} seconds".format(time.time() - start))

        # Run the train function
        # print("Training a random batch ...")
        # start = time.time()
        loss, ec, dc = train(
            input_batches, input_lengths, target_batches, target_lengths,
            encoder, decoder, 
            encoder_optimizer, decoder_optimizer, criterion,
            clip=clip
        )
        # print(epoch, loss)
        # print("Training a random batch took {0:.3f} seconds".format(time.time() - start))

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

            # Validation metrics
            error_val = 0
            all_unigrams = []
            all_bigrams = []
            distinct1 = 0
            distinct2 = 0
            embed_greedy = 0
            embed_avg = 0
            embed_extrema = 0
            affect_distance = 0
            affect_strength = 0
            for i in range(n_validations):
                input_batches, input_lengths, target_batches, target_lengths = random_batch(corpus, pairs_val, batch_size)
                eval_metrics = validate(corpus, word_embedding, affect_embedding_copy, input_batches, input_lengths, target_batches, target_lengths, encoder, decoder)
                # print(eval_metrics)
                error_val += eval_metrics[0]
                all_unigrams += eval_metrics[1]
                all_bigrams += eval_metrics[2]
                embed_greedy += eval_metrics[3]
                embed_avg += eval_metrics[4]
                embed_extrema += eval_metrics[5]
                affect_distance += eval_metrics[6]
                affect_strength += eval_metrics[7]
            
            error_val = error_val/n_validations
            if len(all_unigrams) != 0:
                distinct1 = len(set(all_unigrams))/len(all_unigrams)
            if len(all_bigrams) != 0:
                distinct2 = len(set(all_bigrams))/len(all_bigrams)
            embed_greedy = embed_greedy/n_validations
            embed_avg = embed_avg/n_validations
            embed_extrema = embed_extrema/n_validations
            affect_distance = affect_distance/n_validations
            affect_strength = affect_strength/n_validations
            perplexity_val = math.exp(float(error_val)) if error_val < 300 else float("inf")
            losses_val.append(error_val)
            losses_val_all.append(error_val)

            print_summary = "{0} (Epoch: {1}, Progress: {2:.2f}%) Loss: {3:.2f}, Perplexity: {4:.2f}. Validation ({12:,} samples) Loss: {5:.2f}, Validation Perplexity: {6:.2f}, \
            Distinct-1: {7:.6f}, Distinct-2: {8:.6f}, Embed-greedy: {9:.3f}, Embed-avg: {10:.3f}, Embed-extrema: {11:.3f}, Affect-distance: {13:.3f}, Affect-strength: {14:.3f}".format(
                time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg, perplexity, error_val, perplexity_val, distinct1, distinct2, 
                embed_greedy, embed_avg, embed_extrema, n_validations * batch_size, affect_distance, affect_strength)
            print(print_summary)

            # Save validation metrics file
            with open(val_metrics_file, "a") as f:
                f.write("{0},{1},{2},{3},{4},{5},{6},{7},{8}\n".format(error_val,perplexity_val,distinct1,distinct2,embed_greedy,embed_avg,embed_extrema,affect_distance,affect_strength))

            # Adjust learnig rate based on validation error
            if use_ReduceLROnPlateau:
                encoder_scheduler_on_validation.step(error_val)
                decoder_scheduler_on_validation.step(error_val)

            # Early stopping
            if early_stopping:
                if losses_val[-1] > losses_val[-2] and losses_val[-2] > losses_val[-3] and losses_val[-3] > losses_val[-4]:
                    print("Early stopping: epoch: {0}, training loss: {1:.2f}, validation loss: {2:.2f}".format(epoch, print_loss_avg, error_val))
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

        if epoch % evaluate_every == 0:
            # Evaluate random samples from test set
            evaluate_randomly(corpus, pairs_test, encoder, decoder, MAX_LENGTH)

        if epoch % save_every == 0 and SAVE_MODEL:
            print("Saving model to ", "./saved/{0}/{1}/{2}-{3}.{4}".format(CORPUS, "model", "encoder", FILE_NAME, "pt"))
            torch.save(encoder.state_dict(), "./saved/{0}/{1}/{2}-{3}.{4}".format(CORPUS, "model", "encoder", FILE_NAME, "pt"))
            torch.save(decoder.state_dict(), "./saved/{0}/{1}/{2}-{3}.{4}".format(CORPUS, "model", "decoder", FILE_NAME, "pt"))
