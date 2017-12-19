"""
All helping methods for model training are here
"""
import time
import math
import random
from functools import reduce

import numpy as np
import bottleneck as bn
import torch
from torch.autograd import Variable

from model.corpus import PAD_token, UNK_token, SOS_token, EOS_token
from util.text_to_tensor import sentences2indexes
from util.masked_cross_entropy import masked_cross_entropy
from model.config import USE_CUDA, GPU_ID, teacher_forcing_ratio, beam_size, alpha


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
def train(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, clip=10):
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0

    batch_size = len(input_lengths)
    # print("Batch size: ", batch_size)
    # print("Input lengths: ", input_lengths)
    # print("Target lengths: ", target_lengths)
    # Encoder
    # print("Encoding a random batch ...")
    start = time.time()
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)
    # print("Encoding a random batch took {0:.3f} seconds".format(time.time() - start))

    # Decoder
    # print("Decoding a random batch ...")
    start = time.time()
    decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    max_target_length = max(target_lengths)
    # print("max_target_length: ", max_target_length)
    all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size))

    if USE_CUDA:
        decoder_input = decoder_input.cuda(GPU_ID)
        all_decoder_outputs = all_decoder_outputs.cuda(GPU_ID)
    
    # Run through decoder one by one
    if random.random() <= teacher_forcing_ratio:
        for t in range(max_target_length):
            # print("Decoding at position {0} ...".format(t))
            start = time.time()
            decoder_output, decoder_hidden, decoder_attn = decoder(decoder_input, decoder_hidden, encoder_outputs)
            # print("Decoding at position {0} took  {1}...".format(t, time.time() - start))
            all_decoder_outputs[t] = decoder_output
            decoder_input = target_batches[t] # Next input is current target, fully teacher forcing
    else:
        for t in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attn = decoder(decoder_input, decoder_hidden, encoder_outputs)
            all_decoder_outputs[t] = decoder_output

            # Next input is the predicted word from previous ietration    
            topv, topi = decoder_output.data.topk(1)
            decoder_input = Variable(topi.squeeze(1))
            if USE_CUDA:
                decoder_input = decoder_input.cuda(GPU_ID)
    # print("Decoding a random batch took {0:.3f} seconds".format(time.time() - start))
    # Loss and optimization
    # print("Loss and optimization ...")
    start = time.time()
    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(), # -> batch x seq
        target_batches.transpose(0, 1).contiguous(), # -> batch x seq
        target_lengths
    )
    loss.backward()

    # Clip gradient norms
    ec = torch.nn.utils.clip_grad_norm(encoder.parameters(), clip) 
    dc = torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)

    encoder_optimizer.step()
    decoder_optimizer.step()
    # print("Loss and optimization took {0:.3f} seconds".format(time.time() - start))

    return loss.data[0], ec, dc

# Validate model using validation set
def validate(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder):
    loss = 0
    batch_size = len(input_lengths)
    
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
    
    # Run through decoder one by one
    if random.random() <= teacher_forcing_ratio:
        for t in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attn = decoder(decoder_input, decoder_hidden, encoder_outputs)
            all_decoder_outputs[t] = decoder_output
            decoder_input = target_batches[t] # Next input is current target, fully teacher forcing
    else:
        for t in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attn = decoder(decoder_input, decoder_hidden, encoder_outputs)
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

    return loss.data[0]

# Evaluate single word and produce sentence output
def evaluate(corpus, encoder, decoder, input_seq, max_length):
    # Input sequence
    input_lengths = [len(input_seq)]
    input_seqs = [sentences2indexes(corpus, input_seq)]
    input_batches = Variable(torch.LongTensor(input_seqs), volatile=True).transpose(0, 1)

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
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        # decoder_attentions[idx,:decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

        # Choose top word from output
        topv, topi = decoder_output.data.topk(beam_size, 1, largest=True, sorted=True) # (beam_size, beam_size) or (1, beam_size) for the first time
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
                    node_matrix[i][j] = Node(node_list[i].score * topv[i][j], topi[i][j])
                    node_matrix[i][j].add_parent(node_list[i])
                    # topv[i][j] = node_list[i].score * topv[i][j] # Update topv scores

            # Select top nodes
            node_list.clear()
            flattened_list = reduce(lambda x,y :x+y ,node_matrix)
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

def score_node(node):
    l_term = (((5 + len(get_ancestors_ids(node))) ** alpha) / ((5 + 1) ** alpha))
    return node.score/l_term


'''
# Scorer class for scoring decoder outputs with length normalization
class GlobalScorer(object):
    def __init__(self, alpha):
        self.alpha = alpha

    # Scoring function for decoder outputs
    def score(self, node):
        # Length normalization
        l_term = (((5 + len(beam.next_ys)) ** self.alpha) / ((5 + 1) ** self.alpha))
        return log_probs / l_term


def print_scores(node_matrix):
    print("\n")
    for i in range(len(node_matrix)):
        for j in range(len(node_matrix[i])):
            print(node_matrix[i][j].score, end=', ')
        print("\n")


def print_list_scores(node_list):
    print("\n")
    print(len(node_list))
    for i in range(len(node_list)):
        print(node_list[i].score, end=', ')
    print("\n")



# Beam search
class Beam(object):
    def __init__(self, beam_size, global_scorer, n_best = 1):
        self.beam_size = beam_size

        self.tt = torch.cuda if USE_CUDA else torch

        # Initialize scores for all beams
        self.scores = self.tt.FloatTensor(beam_size).zero_()
        self.all_scores = []

        # The backpointers at each time step
        self.prev_ks = []

        # The outputs at each time step
        self.next_ys = [self.tt.LongTensor(beam_size).fill_(PAD_token)]
        self.next_ys[0][0] = SOS_token
        self.eos_top = False

        self.finished = []
        self.n_best = n_best

        # Global scoring
        self.global_scorer = global_scorer

    # Get current outputs for current time step
    def get_current_state(self):
        return self.next_ys[-1]

    # Get the backpointers for current time step
    def get_current_origin(self):
        return self.prev_ks[-1]

    def advance(self, word_lk):
        num_words = word_lk.size(1)

        # Sum previous scores
        if len(self.prev_ks) > 0:
            beam_lk = word_lk + self.scores.unsqueeze(1).expand_as(word_lk)

            for i in range(self.next_ys[-1]).size(0):
                if self.next_ys[-1][i] == EOS_token:
                    beam_lk[i] = -1e20
        else:
            beam_lk = word_lk[0]
        
        # Top beam_size scores
        best_scores, best_scores_id = beam_lk.view(-1).topk(self.beam_size, 0, True, True)

        self.all_scores.append(self.scores)
        self.scores = best_scores

        # Calculate which word and beam each score came from
        prev_k = best_scores_id/num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append((best_scores_id - prev_k * num_words))

        for i in range(self.next_ys[-1].size(0)):
            if self.next_ys[-1][i] == EOS_token:
                s = self.scores[i]
                global_scores = self.global_scorer.score(self, self.scores)
                s = global_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))

        # End condition is when top-of-beam is EOS and no global score
        if self.next_ys[-1][0] == EOS_token:
            self.eos_top = True
        
    def done(self):
        return self.eos_top and len(self.finished) >= self.n_best

    def sort_finished(self, minimum=None):
        if minimum is not None:
            i = 0
            while len(self.finished) < minimum:
                s = self.scores[i]
                global_scores = self.global_scorer.score(self, self.scores)
                s = global_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))
        self.finished.sort(key=lambda a: -a[0])
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        return scores, ks

    def get_hypothesis(self, timestep, k):
        # Walk back to construct the full hypothesis
        hyp = []

        for j in range(len(self.prev_ks[:timestep]) - 1, -1, -1):
            hyp.append(self.next_ys[j+1][k])
            k = self.prev_ks[j][k]
        return hyp[::-1]

'''

            
