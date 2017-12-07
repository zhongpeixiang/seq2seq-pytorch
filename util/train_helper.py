"""
All helping methods for model training are here
"""
import time
import math

import torch
from torch.autograd import Variable

from model.corpus import PAD_token, UNK_token, SOS_token, EOS_token
from util.process_text import MAX_LENGTH
from util.text_to_tensor import sentences2indexes
from util.masked_cross_entropy import masked_cross_entropy

USE_CUDA = True

# Convert seconds into minutes and seconds
def as_minutes(s):
    m = math.floor(s/60)
    s -= m*60
    return "{0}m {1}s".format(m, s)

# Compute time elapsed since a previous timestamp
def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s/(percent)
    rs = es - s
    return "{0} (- {1})".format(as_minutes(s), as_minutes(rs))

# Train model
def train(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH, clip=10):
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
        decoder_input = decoder_input.cuda()
        all_decoder_outputs = all_decoder_outputs.cuda()
    
    # Run through decoder one by one
    for t in range(max_target_length):
        decoder_output, decoder_hidden, decoder_attn = decoder(decoder_input, decoder_hidden, encoder_outputs)
        all_decoder_outputs[t] = decoder_output
        decoder_input = target_batches[t] # Next input is current target, fully teacher forcing
    
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

# Evaluate single word
def evaluate(corpus, encoder, decoder, input_seq, max_length=MAX_LENGTH):
    input_lengths = [len(input_seq)]
    input_seqs = [sentences2indexes(corpus, input_seq)]
    input_batches = Variable(torch.LongTensor(input_seqs), volatile=True).transpose(0, 1)

    if USE_CUDA:
        input_batches = input_batches.cuda()
    
    # Set to non-training mode to diable dropout
    encoder.train(False)
    decoder.train(False)

    # Encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

    # Decoder
    decoder_input = Variable(torch.LongTensor([SOS_token]), volatile=True) # SOS
    decoder_hidden = encoder_hidden[:decoder.n_layers] # Use last (forward) hidden state from encoder
    
    if USE_CUDA:
        decoder_input = decoder_input.cuda()

    # Store output words and attention states
    decoded_words = []
    decoder_attentions = torch.zeros(max_length + 1, max_length + 1)

    # Run through decoder
    for idx in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        decoder_attentions[di,:decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

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
            decoder_input = decoder_input.cuda()
    
    # Set back to training mode
    encoder.train(True)
    decoder.train(True)

    return decoded_words, decoder_attentions[:di+1, :len(encoder_outputs)]

# Randomly evaluate a pair 
def evaluate_randomly(corpus):
    [input_sentence, target_sentence] = random.choice(pairs)
    output_words, attentions = evaluate(corpus, encoder, decoder, input_sentence)
    output_sentence = " ".join(output_words)
    print('>', input_sentence)
    if target_sentence is not None:
        print('=', target_sentence)
    print('<', output_sentence)
