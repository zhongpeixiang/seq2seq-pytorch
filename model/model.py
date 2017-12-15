"""
Seq2Seq model
Seq2Seq model consists of an encoder and decoder. 
An encoder is a RNN that takes input sequence and transforms it into a fixed-length vector representation
A decoder is another RNN that takes the fixed-length vector representation from encoder and transforms it into a output sequence
The lengths of input sequence and output sequence can be different

Attention mechanism
The fixed-length vector representation is responsible to carry the information of the entire input sequence, which is a huge burden for it
The attention mechanism lets the decoder to focus a part of the input sequence while decoding, instead of relying on the fixed-length vector representation alone
The attention mechanism outputs a context vector, which is the weighted sum of all encoder outputs whose weights are determined by current decoder hidden state 
and each encoder output
The decoder output is computed based on the context vector and current hidden state
"""
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from model.config import MAX_LENGTH, USE_CUDA, GPU_ID


class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

        self.init_weights()

    def forward(self, input_seqs, input_lengths, hidden=None):
        embedded = self.embedding(input_seqs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:] # Sum bidirectional outputs
        return outputs, hidden # (seq_len, batch, hidden_size), (num_layers * num_directions, batch, hidden_size)

    def init_weights(self):
        self.embedding.weight = nn.Parameter(torch.Tensor(np.random.uniform(-0.1, 0.1, (self.input_size, self.embedding_size))))

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if method == 'general':
            self.attn = nn.Linear(hidden_size, hidden_size)

        elif method == 'concat':
            self.attn = nn.Linear(hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))
        
    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)

        attn_energies = Variable(torch.zeros(this_batch_size, max_len)) # (batch_size, max_length)
        if USE_CUDA:
            attn_energies = attn_energies.cuda(GPU_ID)
        
        # For each sample in the batch
        for b in range(this_batch_size):
            # Calculate energy for each encoder output
            for i in range(max_len):
                attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0))

        # Normalize energies to weights in (0, 1), resize to (1, batch_size, max_length)
        return F.softmax(attn_energies).unsqueeze(1)

    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            energy = torch.dot(hidden.view(-1), encoder_output.view(-1))
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = torch.dot(hidden.view(-1), energy.view(-1))
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = torch.dot(self.v.view(-1), energy.view(-1))
        return energy

class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size, max_length=MAX_LENGTH, n_layers=1, dropout_p=0.1):
        super(BahdanauAttnDecoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        # Define layers
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.dropout = nn.Dropout(dropout_p)
        self.attn = Attn('concat', hidden_size)
        self.gru = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size, output_size)

        self.init_weights()

    def forward(self, word_input, last_hidden, encoder_outputs):
        # Get embedding vector of last output word
        word_embedded = self.embedding(word_input).view(1, 1, -1)
        word_embedded = self.dropout(word_embedded)

        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attn(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        context = context.transpose(0, 1)

        # Combine embedded input word and context
        run_input = torch.cat((word_embedded, context), 2)
        output, hidden = self.gru(run_input, last_hidden)

        # Final output layer
        output = output.squeeze(0)
        output = F.log_softmax(self.out(torch.cat((output, context), 1)))

        # Return final output, hidden state and attention weights
        return output, hidden, attn_weights
    
    def init_weights(self):
        self.embedding.weight = nn.Parameter(torch.Tensor(np.random.uniform(-0.1, 0.1, (self.output_size, self.embedding_size))))

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding_size, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        
        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)
        
        self.init_weights()

    def forward(self, input_seq, last_hidden, encoder_outputs):
        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.embedding_size) # S=1 x B x N

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(embedded, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x S=1 x N

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0) # S=1 x B x N -> B x N
        context = context.squeeze(1)       # B x S=1 x N -> B x N
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = F.tanh(self.concat(concat_input))

        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output)

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights

    def init_weights(self):
        self.embedding.weight = nn.Parameter(torch.Tensor(np.random.uniform(-0.1, 0.1, (self.output_size, self.embedding_size))))

