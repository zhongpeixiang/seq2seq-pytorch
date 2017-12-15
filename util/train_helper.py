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
from model.config import MAX_LENGTH, USE_CUDA, GPU_ID, teacher_forcing_ratio, beam_size


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
    
    # Loss and optimization
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
def evaluate(corpus, encoder, decoder, input_seq, max_length=MAX_LENGTH):
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
    """
    for idx in range(max_length):
        # Count how many beams have EOS already
        EOS_count = 0
        for is_eos in is_EOS:
            if is_eos:
                EOS_count += 1
        # print(idx)
        # print(decoder_input.size())
        # print(type(decoder_input))
        # print(decoder_hidden.size())
        # print(type(decoder_hidden))
        # print(encoder_outputs.size())
        # print(type(encoder_outputs))
        print(is_EOS)
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        # decoder_attentions[idx,:decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

        # deocder_output: (beam_size, vocab_size)
        # Choose beam_size of top word from output
        topv, topi = decoder_output.data.topk(beam_size, 1, largest=True, sorted=True) # (beam_size, beam_size) or (1, beam_size) for the first time
        # print(topv.size())
        # print(topi.size())
        if not first:
            # Add mask for EOS
            for i in range(beam_size):
                if is_EOS[i]:
                    prev_ks[i] = 0 # Ignore beams with EOS already
            print(prev_ks)
            topv = torch.FloatTensor(prev_ks.cpu().numpy()[:, np.newaxis] * topv.cpu().numpy()).cuda(GPU_ID) # Multiply with previous probs
        prev_ks = topv.view(-1).topk(beam_size)[0] # Probs for words in beams without EOS
        # print(topv.size())
        # print(prev_ks.size())
        # decoder_input = Variable(topi.view(-1)[topv.view(-1).topk(beam_size)[1]]) # (beam_size, )
        if first:
            decoder_input = decoder_input.repeat(beam_size)
            decoder_hidden = decoder_hidden.repeat(1, beam_size, 1)
            encoder_outputs = encoder_outputs.repeat(1, beam_size, 1)
        
        # Add ids to output word
        top_indexes = top_n_indexes(topv.cpu().numpy(), beam_size) # Select top n indexes whose value are the largest
        top_indexes = sorted(top_indexes, key=lambda a: topv.cpu().numpy()[a], reverse=True) # Sort these indexes in ascending order
        
        new_prev_ids = prev_ids
        for i, top_idx in enumerate(top_indexes):

            # Use words with top probs as input
            decoder_input[i] = topi[top_idx] # (beam_size, )
            
            # If this beam does not contain EOS
            if is_EOS[top_idx[0]] == False:
                new_prev_ids[i] = prev_ids[top_idx[0]]
                new_prev_ids[i][idx + 1] = topi[top_idx]
                
            if topi[top_idx] == EOS_token and is_EOS[top_idx[0]] == False:
                print("Adding decoded beam...")
                print(i, top_idx, topi[top_idx])
                is_EOS[top_idx[0]] = True
                decoded_beams.append(new_prev_ids[i]) # Add this beam with EOS to the list for decoded beams

        prev_ids = new_prev_ids
        if USE_CUDA:
            decoder_input = decoder_input.cuda(GPU_ID)
        first = False

        # Stop decoding if all beams contain EOS
        print("Number of decoded beams: ", len(decoded_beams))
        if len(decoded_beams) == beam_size:
            print("All beams are finished decoding...")
            break
    
    # print(prev_ids)
    # Output words
    for idx in range(len(decoded_beams)):
        decoded_words.append([corpus.index2word[word_id] for word_id in decoded_beams[idx]])
    
    """

    """
    # Run through decoder
    for idx in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        decoder_attentions[idx,:decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(corpus.index2word[ni])
        
        # Next input is chosen top word
        decoder_input = Variable(torch.LongTensor([ni]))
        if USE_CUDA:
            decoder_input = decoder_input.cuda(GPU_ID)
    """
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
def evaluate_randomly(corpus, pairs, encoder, decoder, max_length=MAX_LENGTH):
    [input_sentence, target_sentence] = random.choice(pairs)
    output_words, attentions = evaluate(corpus, encoder, decoder, input_sentence, max_length=max_length)
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
        """
        `word_lk`- probs of advancing from the last step (K x words)
        """
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


# Scorer class for scoring decoder outputs with length normalization
class GlobalScorer(object):
    def __init__(self, alpha):
        self.alpha = alpha

    # Scoring function for decoder outputs
    def score(self, beam, log_probs):
        # Length normalization
        l_term = (((5 + len(beam.next_ys)) ** self.alpha) / ((5 + 1) ** self.alpha))
        return log_probs / l_term
            