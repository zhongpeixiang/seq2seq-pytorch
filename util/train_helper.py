"""
All helping methods for model training are here
"""
import time
import math
import random
from functools import reduce
import string

import numpy as np
from scipy.spatial.distance import cosine
import bottleneck as bn
import torch
from torch.autograd import Variable
from nltk import bigrams

from model.corpus import PAD_token, UNK_token, SOS_token, EOS_token
from util.text_to_tensor import sentences2indexes
from util.masked_cross_entropy import masked_cross_entropy
from model.config import USE_CUDA, GPU_ID, AFFECT_ATTN, AFFECT_LOSS_STRENGTH, teacher_forcing_ratio, beam_size, alpha, attn_decay


# Convert seconds into minutes and seconds
def as_minutes(s):
    m = math.floor(s/60)
    s -= m*60
    return "{0}m {1}s".format(m, int(s))

# Compute time elapsed since a previous timestamp
def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s/(percent)
    rs = es - s
    return "{0} (- {1})".format(as_minutes(s), as_minutes(rs))

# Train model
def train(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, clip=5):
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0
    affect_embedding = None

    batch_size = len(input_lengths)
    # print("Batch size: ", batch_size)
    # print("Input lengths: ", input_lengths)
    # print("Target lengths: ", target_lengths)
    # Encoder
    # print("Encoding a random batch ...")
    # start = time.time()
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None) # input_batches: (max_length, batch_size)
    # print("Encoding a random batch took {0:.3f} seconds".format(time.time() - start))

    # Decoder
    # print("Decoding a random batch ...")
    # start = time.time()
    decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    max_target_length = max(target_lengths)
    # print("max_target_length: ", max_target_length)
    all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size))
    batch_attn = Variable(torch.zeros(batch_size, max_target_length, encoder_outputs.size(0))).cuda(GPU_ID) # (batch_size, max_target_len, max_input_len)

    if USE_CUDA:
        decoder_input = decoder_input.cuda(GPU_ID)
        all_decoder_outputs = all_decoder_outputs.cuda(GPU_ID)
    if AFFECT_ATTN is not None:
        affect_embedding = encoder.embedding(input_batches)[:, :, -3:] # Get affect embedding for current input batch, (max_length, batch_size, 3)
        # print(affect_embedding.size())
        # print("---------------------")
    
    # Run through decoder one by one
    if random.random() <= teacher_forcing_ratio:
        for t in range(max_target_length):
            # print("Decoding at position {0} ...".format(t))
            # start = time.time()
            decoder_output, decoder_hidden, decoder_attn = decoder(decoder_input, decoder_hidden, encoder_outputs, affect_embedding)
            all_decoder_outputs[t] = decoder_output
            decoder_input = target_batches[t] # Next input is current target, fully teacher forcing
            
            # decoder_attn: (batch_size, 1, max_length)
            batch_attn[:,t,:] = decoder_attn.squeeze(1)
    else:
        for t in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attn = decoder(decoder_input, decoder_hidden, encoder_outputs, affect_embedding)
            
            all_decoder_outputs[t] = decoder_output

            # Next input is the predicted word from previous ietration    
            topv, topi = decoder_output.data.topk(1)
            decoder_input = Variable(topi.squeeze(1))
            if USE_CUDA:
                decoder_input = decoder_input.cuda(GPU_ID)
            
            # decoder_attn: (batch_size, 1, max_length)
            batch_attn[:,t,:] = decoder_attn.squeeze(1)
    # print("Decoding a random batch took {0:.3f} seconds".format(time.time() - start))
    # Loss and optimization
    # print("Loss and optimization ...")
    # start = time.time()
    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(), # -> batch x seq
        target_batches.transpose(0, 1).contiguous(), # -> batch x seq
        target_lengths
    ) # Variable type, size: [1]
    perplexity_loss = loss.data[0]

    # Attention loss
    if attn_decay > 0:
        attn_loss = 0
        identity_matrix = Variable(torch.FloatTensor(np.eye(batch_attn.size(1)))).cuda(GPU_ID)
        for idx in range(batch_attn.size(0)):
            attn_matrix = batch_attn[idx]
            attn_loss += torch.norm(attn_matrix.mm(attn_matrix.transpose(0, 1)) - identity_matrix)
        attn_loss = attn_loss/batch_attn.size(0)
        loss += attn_decay * attn_loss

    # Add sentence affect loss
    if AFFECT_LOSS_STRENGTH != 0:
        output_affect_embedding = decoder.embedding(all_decoder_outputs.contiguous().max(2)[1])[:, :, -3:]
        target_affect_embedding = decoder.embedding(target_batches.contiguous())[:, :, -3:]
        affect_loss = torch.dist(output_affect_embedding.mean(dim=0).mean(dim=0), target_affect_embedding.mean(dim=0).mean(dim=0))
        # print(affect_loss.data[0])
        loss = (1 - AFFECT_LOSS_STRENGTH)*loss + AFFECT_LOSS_STRENGTH * affect_loss
    

    loss.backward()

    # Clip gradient norms
    ec = torch.nn.utils.clip_grad_norm(encoder.parameters(), clip) 
    dc = torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)

    encoder_optimizer.step()
    decoder_optimizer.step()
    # print("Loss and optimization took {0:.3f} seconds".format(time.time() - start))

    return perplexity_loss, ec, dc # return perplexity loss


# Calculate greedy similarity between two sentences
def greedy_similarity(sent1, sent2):
    # sent1: a non-empty list of word embedding
    greedy_score = 0
    for embed_1 in sent1:
        max_score = 0
        for embed_2 in sent2:
            score = 1 - cosine(embed_1, embed_2)
            if score > max_score:
                max_score = score
        greedy_score += max_score
    greedy_score = greedy_score/len(sent1)
    return greedy_score

# Calculate average similarity between two sentences
def avg_similarity(sent1, sent2):
    # sent1: a non-empty list of word embedding
    avg_score = 0
    sum_1 = 0
    sum_2 = 0
    for embed in sent1:
        sum_1 += embed
    for embed in sent2:
        sum_2 += embed
    sum_1 = sum_1/len(sent1)
    sum_2 = sum_2/len(sent2)
    avg_score = 1 - cosine(sum_1, sum_2)
    return avg_score

# Calculate average similarity between two sentences
def extrema_similarity(sent1, sent2):
    # sent1: a non-empty list of word embedding
    extrema_score = 0
    matrix1 = np.asarray(sent1)
    matrix2 = np.asarray(sent2)
    extrema_score = 1 - cosine(np.max(matrix1, axis=0), np.max(matrix2, axis=0))
    return extrema_score

def affect_distance(sent1, sent2):
    avg_distance = 0
    sum_1 = 0
    sum_2 = 0
    for embed in sent1:
        sum_1 += embed
    for embed in sent2:
        sum_2 += embed
    sum_1 = sum_1/len(sent1)
    sum_2 = sum_2/len(sent2)
    avg_distance = np.linalg.norm(sum_1 - sum_2)
    return avg_distance

# Calculate embedding similaries for three modes: greedy matching, average, and extrema
def embed_similarity(batch_tokens, target_tokens, output_embedding, target_embedding):
    # batch_tokens, target_tokens: (batch_size, seq_len)
    # output_embedding: (batch_size, seq_len, embedding_size)
    
    # Calculate greedy similarity
    greedy_score = 0
    avg_score = 0
    extrema_score = 0
    for idx in range(output_embedding.size(0)):
        # Iterate through a generated sentence
        sent1 = []
        sent2 = []
        for token_idx, token in enumerate(batch_tokens[idx]):
            if token == EOS_token:
                break
            sent1.append(output_embedding[idx][token_idx].numpy())
        
        # Iterate through a target sentence
        for token_idx, token in enumerate(target_tokens[idx]):
            if token == EOS_token:
                break
            sent2.append(target_embedding[idx][token_idx].numpy())
        
        if len(sent1) != 0 and len(sent2) != 0: 
            greedy_score += 0.5 * (greedy_similarity(sent1, sent2) + greedy_similarity(sent2, sent1))
            avg_score += avg_similarity(sent1, sent2)
            extrema_score += extrema_similarity(sent1, sent2)
    greedy_score = greedy_score/output_embedding.size(0)
    avg_score = avg_score/output_embedding.size(0)
    extrema_score = extrema_score/output_embedding.size(0)

    return greedy_score, avg_score, extrema_score


def affect_sent_strength(sent):
    avg_strength = 0
    matrix = np.asarray(sent)
    avg_strength = np.linalg.norm(matrix)/len(sent)
    return avg_strength

def affect_metrics(batch_tokens, target_tokens, output_affect_embedding, target_affect_embedding):
    # batch_tokens, target_tokens: (batch_size, seq_len)
    # output_affect_embedding: (batch_size, seq_len, 3)
    
    # Calculate cosine similarity
    affect_dist = 0
    affect_strength = 0
    for idx in range(output_affect_embedding.size(0)):
        # Iterate through a generated sentence
        sent1 = []
        sent2 = []
        for token_idx, token in enumerate(batch_tokens[idx]):
            if token == EOS_token:
                break
            sent1.append(output_affect_embedding[idx][token_idx].numpy())
        
        # Iterate through a target sentence
        for token_idx, token in enumerate(target_tokens[idx]):
            if token == EOS_token:
                break
            sent2.append(target_affect_embedding[idx][token_idx].numpy())
        
        if len(sent1) != 0 and len(sent2) != 0: 
            affect_dist += affect_distance(sent1, sent2)
            affect_strength += affect_sent_strength(sent1)

    affect_dist = affect_dist/output_affect_embedding.size(0)
    affect_strength = affect_strength/output_affect_embedding.size(0)
    
    return affect_dist, affect_strength


# Validate model using validation set
def validate(corpus, word_embedding, affect_embedding_copy, input_batches, input_lengths, target_batches, target_lengths, encoder, decoder):
    loss = 0
    batch_size = len(input_lengths)
    affect_embedding = None
    
    # Encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

    # Decoder
    decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    max_target_length = max(target_lengths)
    all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size))

    if USE_CUDA:
        decoder_input = decoder_input.cuda(GPU_ID)
        all_decoder_outputs = all_decoder_outputs.cuda(GPU_ID)
    if AFFECT_ATTN is not None:
        affect_embedding = encoder.embedding(input_batches)[:, :, -3:] # Get affect embedding for current input batch, (max_length, batch_size, 3)
    # Run through decoder one by one
    if random.random() <= teacher_forcing_ratio:
        for t in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attn = decoder(decoder_input, decoder_hidden, encoder_outputs, affect_embedding)
            all_decoder_outputs[t] = decoder_output
            decoder_input = target_batches[t] # Next input is current target, fully teacher forcing
    else:
        for t in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attn = decoder(decoder_input, decoder_hidden, encoder_outputs, affect_embedding)
            all_decoder_outputs[t] = decoder_output

            # Next input is the predicted word from previous ietration    
            topv, topi = decoder_output.data.topk(1)
            decoder_input = Variable(topi.squeeze(1))
            if USE_CUDA:
                decoder_input = decoder_input.cuda(GPU_ID)
    # Loss
    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(), # -> batch x seq
        target_batches.transpose(0, 1).contiguous(), # -> batch x seq
        target_lengths
    )
    perplexity_loss = loss.data[0]

    if AFFECT_LOSS_STRENGTH != 0:
        output_affect_embedding = decoder.embedding(all_decoder_outputs.contiguous().max(2)[1])[:, :, -3:]
        target_affect_embedding = decoder.embedding(target_batches.contiguous())[:, :, -3:]
        affect_loss = torch.dist(output_affect_embedding.mean(dim=0).mean(dim=0), target_affect_embedding.mean(dim=0).mean(dim=0))
        loss = (1 - AFFECT_LOSS_STRENGTH)*loss + AFFECT_LOSS_STRENGTH * affect_loss

    # Distinct-1 and distinct-2
    batch_tokens = all_decoder_outputs.transpose(0, 1).contiguous().max(2)[1].data # batch_tokens: (batch_size, seq_len)
    target_tokens = target_batches.transpose(0, 1).contiguous() # (batch_size, seq_len)
    # print(batch_tokens)
    batch_unigrams = []
    batch_bigrams = []
    for idx in range(batch_tokens.size(0)):
        tokens = []
        for token in batch_tokens[idx]:
            if token == EOS_token:
                break
            if corpus.index2word[token] not in string.punctuation:
                tokens.append(token)
        bi_grams = [item for item in bigrams(tokens)]
        batch_unigrams += tokens
        batch_bigrams += bi_grams
    
    # Embedding Metrics
    output_embedding = word_embedding(all_decoder_outputs.transpose(0, 1).contiguous().max(2)[1]) # output_embedding: (batch_size, seq, embedding_size)
    target_embedding = word_embedding(target_batches.transpose(0, 1).contiguous()) # target_embedding: (batch_size, seq, embedding_size)
    embed_greedy, embed_avg, embed_extrema = embed_similarity(batch_tokens.cpu(), target_tokens.cpu().data, output_embedding.cpu().data, target_embedding.cpu().data)

    # Affect Embedding Metrics
    output_affect_embedding = affect_embedding_copy(all_decoder_outputs.transpose(0, 1).contiguous().max(2)[1]) # output_affect_embedding: (batch_size, seq, 3)
    target_affect_embedding = affect_embedding_copy(target_batches.transpose(0, 1).contiguous()) # target_affect_embedding: (batch_size, seq, 3)
    affect_similarity, affect_strength = affect_metrics(batch_tokens.cpu(), target_tokens.cpu().data, output_affect_embedding.cpu().data, target_affect_embedding.cpu().data)

    return perplexity_loss, batch_unigrams, batch_bigrams, embed_greedy, embed_avg, embed_extrema, affect_similarity, affect_strength

# Evaluate single word and produce sentence output
def evaluate(corpus, encoder, decoder, input_seq, max_length):
    # Input sequence
    input_lengths = [len(input_seq)]
    input_seqs = [sentences2indexes(corpus, input_seq)]
    input_batches = Variable(torch.LongTensor(input_seqs), volatile=True).transpose(0, 1)
    affect_embedding = None

    if USE_CUDA:
        input_batches = input_batches.cuda(GPU_ID)
    
    # Set to non-training mode to diable dropout
    encoder.train(False)
    decoder.train(False)

    # Encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

    # Decoder
    decoder_input = Variable(torch.LongTensor([SOS_token]), volatile=True) # SOS, (beam_size, )
    decoder_hidden = encoder_hidden[:decoder.n_layers] # Use last (forward) hidden state from encoder
    
    if USE_CUDA:
        decoder_input = decoder_input.cuda(GPU_ID)

    # Store output words and attention states
    decoded_words = []
    decoder_attentions = torch.zeros(max_length + 1, max_length + 1)

    # Beam search
    first = True
    node_list = []
    node_matrix = [[None] * beam_size for _ in range(beam_size)]
    finished_nodes = []

    for idx in range(max_length):
        # Affect embedding, to change affect_embedding size
        if AFFECT_ATTN is not None:
            if first:
                affect_embedding = encoder.embedding(input_batches)[:, :, -3:] # Get affect embedding for current input batch, (max_length, batch_size, 3)
            else:
                affect_embedding = encoder.embedding(input_batches.repeat(1, beam_size))[:, :, -3:]
        # Decoder
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs, affect_embedding
        )
        # decoder_attentions[idx,:decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

        # Choose top word from output
        topv, topi = decoder_output.data.topk(beam_size, 1, largest=True, sorted=True) # (beam_size, beam_size) or (1, beam_size) for the first time
        topv = torch.log(topv) # take log of softmax probabilities
        if first:
            decoder_input = decoder_input.repeat(beam_size)
            decoder_hidden = decoder_hidden.repeat(1, beam_size, 1)
            encoder_outputs = encoder_outputs.repeat(1, beam_size, 1)

            for i in range(beam_size):
                if topi[0][i] == EOS_token:
                    topi[0][i] = SOS_token
                node_list.append(Node(topv[0][i], topi[0][i]))
        else:
            # Build a matrix of nodes, size: (beam_size, beam_size)
            for i in range(beam_size):
                for j in range(beam_size):
                    ancestors = get_ancestors_ids(node_list[i])
                    beam_length = 0
                    for e in ancestors:
                        beam_length += 1
                    length_term_before = node_length_term(beam_length)
                    length_term_after = node_length_term(beam_length + 1)

                    # Added  length penalty
                    node_matrix[i][j] = Node((node_list[i].score + topv[i][j]) * length_term_before/length_term_after, topi[i][j])
                    node_matrix[i][j].add_parent(node_list[i])

            # Select top nodes
            node_list.clear()
            flattened_list = reduce(lambda x,y :x+y , node_matrix)
            sorted_flattened_list = sorted(flattened_list, key=lambda node: node.score, reverse=True)
            for node in sorted_flattened_list:
                if node.word_id != EOS_token:
                    node_list.append(node)
                    if len(node_list) == beam_size:
                        break
                else:
                    finished_nodes.append(node)
                    if len(finished_nodes) == beam_size:
                        break
                        
        if len(finished_nodes) == beam_size:
            break
        
        # Next inputs are all nodes in the node list
        for i, node in enumerate(node_list):
            decoder_input[i] = node.word_id
        if USE_CUDA:
            decoder_input = decoder_input.cuda(GPU_ID)

        first = False
    
    # Get ancestor ids
    for node in finished_nodes:
        decoded_words.append([corpus.index2word[word_id] for word_id in get_ancestors_ids(node)])
    
    # Set back to training mode
    encoder.train(True)
    decoder.train(True)

    return decoded_words, decoder_attentions[:idx+1, :len(encoder_outputs)]

# Randomly evaluate a pair from a corpus
def evaluate_randomly(corpus, pairs, encoder, decoder, max_length):
    [input_sentence, target_sentence] = random.choice(pairs)
    output_words, attentions = evaluate(corpus, encoder, decoder, input_sentence, max_length)
    output_sentences = [" ".join(output_sent) for output_sent in output_words]
    print('>', input_sentence)
    if target_sentence is not None:
        print('=', target_sentence)
    for output_sentence in output_sentences:
        print('<', output_sentence)

# Top n indexes from a numpy array
def top_n_indexes(arr, n):
    idx = bn.argpartition(arr, arr.size-n, axis=None)[-n:]
    width = arr.shape[1]
    return [divmod(i, width) for i in idx]


class Node(object):
    def __init__(self, score, word_id):
        self.score = score
        self.word_id = word_id
        self.parent = None

    def add_parent(self, obj):
        self.parent = obj       


def get_ancestors_ids(node):
    word_ids = [node.word_id]
    while(node.parent != None):
        word_ids.append(node.parent.word_id)
        node = node.parent
    return reversed(word_ids)

def node_length_term(node_length):
    l_term = (((5 + node_length) ** alpha) / ((5 + 1) ** alpha))
    return l_term

