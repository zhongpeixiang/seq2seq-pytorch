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
import torch.nn.init as weight_init

from model.config import USE_CUDA, GPU_ID, LOAD_WORD2VEC, AFFECT_EMBEDDING_STRENGTH


class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=1, dropout=0.1, embedding=None):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.embedding.weight = nn.Parameter(torch.Tensor(np.random.uniform(-0.1, 0.1, (input_size, embedding_size))))
        # If use word2vec embedding
        if embedding is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding))
        
        self.gru = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

        print("Initializing encoder weights...")
        for name, param in self.named_parameters(): 
            if len(param.data.size()) < 2:
                nn.init.uniform(param.data, 1/4, 1/2)
            elif "embedding" not in name:
                nn.init.xavier_uniform(param.data)

    def forward(self, input_seqs, input_lengths, hidden=None):
        embedded = self.embedding(input_seqs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:] # Sum bidirectional outputs
        return outputs, hidden # (seq_len, batch, hidden_size), (num_layers * num_directions, batch, hidden_size)

class Attn(nn.Module):
    def __init__(self, method, hidden_size, affect_attn):
        super(Attn, self).__init__()

        self.affect_attn = affect_attn
        self.method = method
        self.hidden_size = hidden_size

        if affect_attn == "affect_attn_matrix" or affect_attn == "affect_attn_matrix_norm":
            self.affect_matrix = nn.Parameter(torch.FloatTensor(np.random.uniform(-0.01, 0.01, (hidden_size, 3))))
        
        if affect_attn == "affect_attn_norm":
            self.affect_matrix = nn.Parameter(torch.FloatTensor(np.random.uniform(-0.1, 0.1, (1, 3))))

        if method == 'general':
            self.attn = nn.Linear(hidden_size, hidden_size)

        elif method == 'concat':
            self.attn = nn.Linear(hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))
        
        print("Initializing attention weights...")
        for name, param in self.named_parameters(): 
            if len(param.data.size()) < 2:
                nn.init.uniform(param.data, 1/4, 1/2)
            elif "embedding" not in name:
                nn.init.xavier_uniform(param.data)

        
    def forward(self, hidden, encoder_outputs, embedding):
        # embedding: input affect embedding (n_words, 3)
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)

        attn_energies = Variable(torch.zeros(this_batch_size, max_len)) # (batch_size, max_length)
        if USE_CUDA:
            attn_energies = attn_energies.cuda(GPU_ID)
        
        if embedding is None:
            # For each sample in the batch
            for b in range(this_batch_size):
                # Calculate energy for each encoder output
                for i in range(max_len):
                    attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0), None)
        else:
            # For each sample in the batch
            # print(hidden.size())
            # print(encoder_outputs.size())
            # print(embedding.size())
            for b in range(this_batch_size):
                # Calculate energy for each encoder output
                for i in range(max_len):
                    # print(embedding[i, b].size())
                    # print(embedding[i, b].unsqueeze(0).size())
                    # print("---------------------")
                    attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0), embedding[i, b].unsqueeze(0))
                   
        # Normalize energies to weights in (0, 1), resize to (1, batch_size, max_length)
        return F.softmax(attn_energies).unsqueeze(1)

    def score(self, hidden, encoder_output, embedding):
        # hidden: (1, 1024)
        # encoder_output: (1, 1024)
        # self.affect_matrix: (1024, 3)
        # embedding: (1, 3)
        if self.affect_attn is not None:
            if self.method == 'dot':
                # Added affective attention
                # print(hidden.size())
                # print(hidden.view(1, -1).size())
                # print(self.affect_matrix.size())
                # print(embedding.size())
                # print(embedding.view(-1, 1).size())
                # print("---------------------")
                if self.affect_attn == "affect_attn_matrix":
                    energy = torch.dot(hidden.view(-1), encoder_output.view(-1)) + (hidden.view(1, -1).mm(self.affect_matrix)).mm(embedding.view(-1, 1))[0][0]
                elif self.affect_attn == "affect_atten_norm":
                    energy = torch.dot(hidden.view(-1), encoder_output.view(-1)) + torch.norm(self.affect_matrix * embedding.view(1, -1))
                elif self.affect_attn == "affect_attn_matrix_norm":
                    energy = torch.dot(hidden.view(-1), encoder_output.view(-1)) + torch.norm((hidden.view(1, -1).mm(self.affect_matrix)) * embedding.view(1, -1))
            elif self.method == 'general':
                energy = self.attn(encoder_output)
                energy = torch.dot(hidden.view(-1), energy.view(-1))
            elif self.method == 'concat':
                energy = self.attn(torch.cat((hidden, encoder_output), 1))
                energy = torch.dot(self.v.view(-1), energy.view(-1))
        else:
            if self.method == 'dot':
                energy = torch.dot(hidden.view(-1), encoder_output.view(-1))
            elif self.method == 'general':
                energy = self.attn(encoder_output)
                energy = torch.dot(hidden.view(-1), energy.view(-1))
            elif self.method == 'concat':
                energy = self.attn(torch.cat((hidden, encoder_output), 1))
                energy = torch.dot(self.v.view(-1), energy.view(-1))
        return energy


class BigramAttn(nn.Module):
    def __init__(self, method, hidden_size, affect_attn):
        super(BigramAttn, self).__init__()

        self.affect_attn = affect_attn
        self.method = method
        self.hidden_size = hidden_size

        self.bigram_matrix = nn.Parameter(torch.FloatTensor(np.random.uniform(-0.01, 0.01, (hidden_size, hidden_size))))

        if affect_attn == "affect_attn_matrix" or affect_attn == "affect_attn_matrix_norm":
            self.affect_matrix = nn.Parameter(torch.FloatTensor(np.random.uniform(-0.01, 0.01, (hidden_size, 3))))

        if affect_attn == "affect_attn_norm":
            self.affect_matrix = nn.Parameter(torch.FloatTensor(np.random.uniform(-0.1, 0.1, (1, 3))))

        if method == 'general':
            self.attn = nn.Linear(hidden_size, hidden_size)

        elif method == 'concat':
            self.attn = nn.Linear(hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))
        
        print("Initializing attention weights...")
        for name, param in self.named_parameters(): 
            if len(param.data.size()) < 2:
                nn.init.uniform(param.data, 1/4, 1/2)
            elif "embedding" not in name:
                nn.init.xavier_uniform(param.data)

        
    def forward(self, hidden, encoder_outputs, embedding):
        # embedding: input affect embedding (n_words, 3)
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)

        attn_energies = Variable(torch.zeros(this_batch_size, max_len)) # (batch_size, max_length)
        if USE_CUDA:
            attn_energies = attn_energies.cuda(GPU_ID)
        
        if embedding is None:
            # For each sample in the batch
            for b in range(this_batch_size):
                # Calculate energy for each encoder output
                for i in range(max_len):
                    if i == 0:
                        attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0), None, None)
                    else:
                        attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0), encoder_outputs[i-1, b].unsqueeze(0), None)
        else:
            for b in range(this_batch_size):
                # Calculate energy for each encoder output
                for i in range(max_len):
                    if i == 0:
                        attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0), None, embedding[i, b].unsqueeze(0))
                    else:
                        attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0), encoder_outputs[i-1, b].unsqueeze(0), embedding[i, b].unsqueeze(0))
                   
        # Normalize energies to weights in (0, 1), resize to (1, batch_size, max_length)
        return F.softmax(attn_energies).unsqueeze(1)

    def score(self, hidden, encoder_output, prev_encoder_output, embedding):
        # hidden: (1, 1024)
        # encoder_output: (1, 1024)
        # prev_encoder_output: (1, 1024) or None
        # self.affect_matrix: (1024, 3)
        # embedding: (1, 3)
        if self.affect_attn is not None:
            if self.method == 'dot':
                if prev_encoder_output is None:
                    bigram_hidden = encoder_output
                else:
                    bigram_hidden = prev_encoder_output.mm(self.bigram_matrix) * encoder_output
                
                if self.affect_attn == "affect_attn_matrix":
                    energy = torch.dot(hidden.view(-1), bigram_hidden.view(-1)) + (hidden.view(1, -1).mm(self.affect_matrix)).mm(embedding.view(-1, 1))[0][0]
                elif self.affect_attn == "affect_atten_norm":
                    energy = torch.dot(hidden.view(-1), bigram_hidden.view(-1)) + torch.norm(self.affect_matrix * embedding)
                elif self.affect_attn == "affect_attn_matrix_norm":
                    energy = torch.dot(hidden.view(-1), bigram_hidden.view(-1)) + torch.norm((hidden.view(1, -1).mm(self.affect_matrix)) * embedding.view(1, -1))
            elif self.method == 'general':
                energy = self.attn(encoder_output)
                energy = torch.dot(hidden.view(-1), energy.view(-1))
            elif self.method == 'concat':
                energy = self.attn(torch.cat((hidden, encoder_output), 1))
                energy = torch.dot(self.v.view(-1), energy.view(-1))
        else:
            if self.method == 'dot':
                if prev_encoder_output is None:
                    bigram_hidden = encoder_output
                else:
                    bigram_hidden = prev_encoder_output.mm(self.bigram_matrix) * encoder_output
                energy = torch.dot(hidden.view(-1), bigram_hidden.view(-1))
            elif self.method == 'general':
                energy = self.attn(encoder_output)
                energy = torch.dot(hidden.view(-1), energy.view(-1))
            elif self.method == 'concat':
                energy = self.attn(torch.cat((hidden, encoder_output), 1))
                energy = torch.dot(self.v.view(-1), energy.view(-1))
        return energy


class BigramConcatAttn(nn.Module):
    def __init__(self, method, hidden_size, affect_attn):
        super(BigramConcatAttn, self).__init__()

        self.affect_attn = affect_attn
        self.method = method
        self.hidden_size = hidden_size

        self.bigram_matrix = nn.Parameter(torch.FloatTensor(np.random.uniform(-0.01, 0.01, (2*hidden_size, hidden_size))))

        if affect_attn == "affect_attn_matrix" or affect_attn == "affect_attn_matrix_norm":
            self.affect_matrix = nn.Parameter(torch.FloatTensor(np.random.uniform(-0.01, 0.01, (hidden_size, 3))))

        if affect_attn == "affect_attn_norm":
            self.affect_matrix = nn.Parameter(torch.FloatTensor(np.random.uniform(-0.1, 0.1, (1, 3))))

        if method == 'general':
            self.attn = nn.Linear(hidden_size, hidden_size)

        elif method == 'concat':
            self.attn = nn.Linear(hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))
        
        print("Initializing attention weights...")
        for name, param in self.named_parameters(): 
            if len(param.data.size()) < 2:
                nn.init.uniform(param.data, 1/4, 1/2)
            elif "embedding" not in name:
                nn.init.xavier_uniform(param.data)

        
    def forward(self, hidden, encoder_outputs, embedding):
        # embedding: input affect embedding (n_words, 3)
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)

        attn_energies = Variable(torch.zeros(this_batch_size, max_len)) # (batch_size, max_length)
        if USE_CUDA:
            attn_energies = attn_energies.cuda(GPU_ID)
        
        # print(attn_energies.size(), hidden.size(), encoder_outputs.size(), embedding.size())
        if embedding is None:
            # For each sample in the batch
            for b in range(this_batch_size):
                # Calculate energy for each encoder output
                for i in range(max_len):
                    if i == 0:
                        attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0), None, None)
                    else:
                        attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0), encoder_outputs[i-1, b].unsqueeze(0), None)
        else:
            for b in range(this_batch_size):
                # Calculate energy for each encoder output
                for i in range(max_len):
                    if i == 0:
                        attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0), None, embedding[i, b].unsqueeze(0))
                    else:
                        attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0), encoder_outputs[i-1, b].unsqueeze(0), embedding[i, b].unsqueeze(0))
                   
        # Normalize energies to weights in (0, 1), resize to (batch_size, 1, max_length)
        return F.softmax(attn_energies).unsqueeze(1)

    def score(self, hidden, encoder_output, prev_encoder_output, embedding):
        # hidden: (1, 1024)
        # encoder_output: (1, 1024)
        # prev_encoder_output: (1, 1024) or None
        # self.affect_matrix: (1024, 3)
        # embedding: (1, 3)
        if self.affect_attn is not None:
            if self.method == 'dot':
                if prev_encoder_output is None:
                    bigram_hidden = encoder_output
                else:
                    bigram_hidden = torch.cat((prev_encoder_output, encoder_output), 1).mm(self.bigram_matrix)
                if self.affect_attn == "affect_attn_matrix":
                    energy = torch.dot(hidden.view(-1), bigram_hidden.view(-1)) + (hidden.view(1, -1).mm(self.affect_matrix)).mm(embedding.view(-1, 1))[0][0]
                elif self.affect_attn == "affect_attn_norm":
                    energy = torch.dot(hidden.view(-1), bigram_hidden.view(-1)) + torch.norm(self.affect_matrix * embedding)
                elif self.affect_attn == "affect_attn_matrix_norm":
                    energy = torch.dot(hidden.view(-1), bigram_hidden.view(-1)) + torch.norm((hidden.view(1, -1).mm(self.affect_matrix)) * embedding.view(1, -1))
            elif self.method == 'general':
                energy = self.attn(encoder_output)
                energy = torch.dot(hidden.view(-1), energy.view(-1))
            elif self.method == 'concat':
                energy = self.attn(torch.cat((hidden, encoder_output), 1))
                energy = torch.dot(self.v.view(-1), energy.view(-1))
        else:
            if self.method == 'dot':
                if prev_encoder_output is None:
                    bigram_hidden = encoder_output
                else:
                    bigram_hidden = torch.cat((prev_encoder_output, encoder_output), 1).mm(self.bigram_matrix)
                energy = torch.dot(hidden.view(-1), bigram_hidden.view(-1))
            elif self.method == 'general':
                energy = self.attn(encoder_output)
                energy = torch.dot(hidden.view(-1), energy.view(-1))
            elif self.method == 'concat':
                energy = self.attn(torch.cat((hidden, encoder_output), 1))
                energy = torch.dot(self.v.view(-1), energy.view(-1))
        return energy


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, bigram_attn, embedding_size, hidden_size, output_size, n_layers=1, dropout=0.1, embedding=None, affect_attn=None):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.bigram_attn = bigram_attn
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.affect_attn = affect_attn

        # Define layers
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.embedding.weight = nn.Parameter(torch.Tensor(np.random.uniform(-0.1, 0.1, (output_size, embedding_size))))
        # If use word2vec embedding
        if embedding is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding))
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        
        # Choose attention model
        if bigram_attn == "BigramAttn" and attn_model != 'none':
            self.attn = BigramAttn(attn_model, hidden_size, affect_attn)
        elif bigram_attn == "BigramConcatAttn" and attn_model != 'none':
            self.attn = BigramConcatAttn(attn_model, hidden_size, affect_attn)
        elif attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size, affect_attn)
        
        print("Initializing decoder weights...")
        for name, param in self.named_parameters():
            if len(param.data.size()) < 2:
                nn.init.uniform(param.data, 1/4, 1/2)
            elif "embedding" not in name:
                nn.init.xavier_uniform(param.data)

    def forward(self, input_seq, last_hidden, encoder_outputs, affect_embedding=None):
        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.embedding_size) # S=1 x B x N

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(embedded, last_hidden) # output (seq_len, batch, hidden_size * num_directions), h_n (num_layers * num_directions, batch, hidden_size)

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(rnn_output, encoder_outputs, affect_embedding) # affect_embedding: input embedding (n_words, 3)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x S=1 x N

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0) # S=1 x B x N -> B x N
        context = context.squeeze(1)       # B x S=1 x N -> B x N
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = F.tanh(self.concat(concat_input))

        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output) # output: (batch_size, output_size)

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights


class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, bigram_attn, embedding_size, hidden_size, output_size, n_layers=1, dropout_p=0.1, embedding=None, affect_attn=None):
        super(BahdanauAttnDecoderRNN, self).__init__()

        self.bigram_attn = bigram_attn
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.affect_attn = affect_attn

        # Define layers
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.embedding.weight = nn.Parameter(torch.Tensor(np.random.uniform(-0.1, 0.1, (output_size, embedding_size))))
        # If use word2vec embedding
        if embedding is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding))
        self.dropout = nn.Dropout(dropout_p)
        if bigram_attn:
            self.attn = BigramAttn('concat', hidden_size, affect_attn)
        else:
            self.attn = Attn('concat', hidden_size, affect_attn)
        self.gru = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size, output_size)

        print("Initializing attention weights...")
        for name, param in self.named_parameters(): 
            if len(param.data.size()) < 2:
                nn.init.uniform(param.data, 1/4, 1/2)
            elif "embedding" not in name:
                nn.init.xavier_uniform(param.data)


    def forward(self, word_input, last_hidden, encoder_outputs, affect_embedding=None):
        # Get embedding vector of last output word
        word_embedded = self.embedding(word_input).view(1, 1, -1)
        word_embedded = self.dropout(word_embedded)

        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attn(last_hidden[-1], encoder_outputs, affect_embedding)
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


# RNN model to predict sentiment (VAD) of sentences
class SentimentRNN(nn.Module):
    def __init__(self, model, input_size, embedding_size, hidden_size, output_size, n_layers, dropout, embedding, freeze_embed, output_projection, use_attention):
        super(SentimentRNN, self).__init__()

        self.model = model
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.freeze_embed = freeze_embed
        self.output_projection = output_projection
        self.use_attention = use_attention
        if use_attention == "self-attention":
            self.attn_size = 200
            self.attn_matrix = nn.Parameter(torch.Tensor(np.random.uniform(-0.01, 0.01, (self.attn_size, 2 * hidden_size)))) # (d, 2 * hidden_size)
            self.attn_vector = nn.Parameter(torch.Tensor(np.random.uniform(-0.01, 0.01, (self.attn_size)))) # (d, )
        elif use_attention == "vec-attention":
            self.attn_vector = nn.Parameter(torch.Tensor(np.random.uniform(-0.01, 0.01, (2 * hidden_size)))) # (2 * hidden_size, )

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.embedding.weight = nn.Parameter(torch.Tensor(np.random.uniform(-0.1, 0.1, (input_size, embedding_size))))
        self.embedding.weight.data.copy_(torch.from_numpy(embedding))
        if freeze_embed:
            self.embedding.weight.requires_grad = False
        
        if model == "rnn":
            self.rnn = nn.RNN(embedding_size, hidden_size, n_layers, dropout=dropout[1], bidirectional=True)
        elif model == "gru":
            self.rnn = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout[1], bidirectional=True)
        elif model == "lstm":
            self.rnn = nn.LSTM(embedding_size, hidden_size, n_layers, dropout=dropout[1], bidirectional=True)
        for name, param in self.rnn.named_parameters(): 
            weight_init.uniform(param, -0.01, 0.01)
        
        
        if output_projection:
            self.linear = nn.Linear(hidden_size * 2, 256)
            self.relu = nn.ReLU()
            self.out = nn.Linear(256, output_size)
        else:
            self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input_seqs, input_lengths):
        batch_size = input_seqs.size(1)
        embedded = self.embedding(input_seqs)
        embedded = F.dropout(embedded, self.dropout[0], training=self.training) # Dropout for embedding
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        hidden = self.init_hidden(batch_size)
        if self.model == "lstm":
            outputs, hidden = self.rnn(packed, hidden) # outputs: (seq_len, batch, hidden_size * num_directions), hidden: (n_layers * n_directions, batch, hidden_size)
            hidden = hidden[0] # Extract hidden states, leave out cell state
        elif self.model == "rnn" or self.model == "gru":
            outputs, hidden = self.rnn(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # hidden = 0.5 * (hidden[self.n_layers - 1] + hidden[-1]) # Average over forward and backward hidden state at top layer
        hidden = torch.cat((hidden[self.n_layers - 1], hidden[-1]), 1) # Concatenate forward and backward hidden state at top layer
        if self.use_attention:
            hidden = self.attention(self.use_attention, outputs) # (batch, 2 * hidden_size)
        hidden = F.dropout(hidden, self.dropout[2], training=self.training) # Dropout for hidden state
        
        # Output
        if self.output_projection:
            hidden = self.linear(hidden)
            relu = self.relu(hidden)
            out = self.out(relu)
        else:
            out = self.out(hidden)
        return out
    
    def attention(self, attention_model, hidden_states):
        # Transform to size: (batch, seq_len, hidden_size * num_directions)
        hidden_states = hidden_states.view(hidden_states.size(1), hidden_states.size(0), hidden_states.size(2))
        if attention_model == "self-attention":
            nonlinear_states = F.tanh(hidden_states.matmul(self.attn_matrix.transpose(0, 1))) # (batch, seq_len, d)
            attn_weights = F.softmax(nonlinear_states.matmul(self.attn_vector)) # (batch, seq_len)
        elif attention_model == "vec-attention":
            nonlinear_states = F.tanh(hidden_states.matmul(self.attn_vector)) # (batch, seq_len)
            attn_weights = F.softmax(nonlinear_states) # (batch, seq_len)
        attn_weights = attn_weights.view(attn_weights.size(0), 1, attn_weights.size(1)) # (batch, 1, seq_len)
        weighted_hidden = attn_weights.bmm(hidden_states)# (batch, 1, seq_len), (batch, seq_len, 2*hidden_size) => (batch, 1, 2*hidden_size)
        return weighted_hidden


    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if self.model == 'lstm':
            return (Variable(weight.new(2 * self.n_layers, batch_size, self.hidden_size).zero_()),
                    Variable(weight.new(2 * self.n_layers, batch_size, self.hidden_size).zero_()))
        else:
            return Variable(weight.new(2 * self.n_layers, batch_size, self.hidden_size).zero_())

# RNN model to predict sentiment (VAD) of sentences
class KaggleSentimentRNN(nn.Module):
    def __init__(self, model, input_size, embedding_size, hidden_size, output_size, n_layers, dropout, embedding, freeze_embed, output_projection, use_attention):
        super(KaggleSentimentRNN, self).__init__()

        self.model = model
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.freeze_embed = freeze_embed
        self.output_projection = output_projection
        self.use_attention = use_attention
        if use_attention == "self-attention":
            self.attn_size = 200
            self.attn_matrix = nn.Parameter(torch.Tensor(np.random.uniform(-0.01, 0.01, (self.attn_size, 2 * hidden_size)))) # (d, 2 * hidden_size)
            self.attn_vector = nn.Parameter(torch.Tensor(np.random.uniform(-0.01, 0.01, (self.attn_size)))) # (d, )
        elif use_attention == "vec-attention":
            self.attn_vector = nn.Parameter(torch.Tensor(np.random.uniform(-0.01, 0.01, (2 * hidden_size)))) # (2 * hidden_size, )

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.embedding.weight = nn.Parameter(torch.Tensor(np.random.uniform(-0.1, 0.1, (input_size, embedding_size))))
        self.embedding.weight.data.copy_(torch.from_numpy(embedding))
        if freeze_embed:
            self.embedding.weight.requires_grad = False
        
        if model == "rnn":
            self.rnn = nn.RNN(embedding_size, hidden_size, n_layers, dropout=dropout[1], bidirectional=True)
        elif model == "gru":
            self.rnn = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout[1], bidirectional=True)
        elif model == "lstm":
            self.rnn = nn.LSTM(embedding_size, hidden_size, n_layers, dropout=dropout[1], bidirectional=True)
        for name, param in self.rnn.named_parameters(): 
            weight_init.uniform(param, -0.01, 0.01)
        
        
        if output_projection:
            self.linear = nn.Linear(hidden_size * 2, 256)
            self.relu = nn.ReLU()
            self.out = nn.Linear(256, output_size)
        else:
            self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input_seqs, input_lengths):
        batch_size = input_seqs.size(1)
        embedded = self.embedding(input_seqs)
        embedded = F.dropout(embedded, self.dropout[0], training=self.training) # Dropout for embedding
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        hidden = self.init_hidden(batch_size)
        if self.model == "lstm":
            outputs, hidden = self.rnn(packed, hidden) # outputs: (seq_len, batch, hidden_size * num_directions), hidden: (n_layers * n_directions, batch, hidden_size)
            hidden = hidden[0] # Extract hidden states, leave out cell state
        elif self.model == "rnn" or self.model == "gru":
            outputs, hidden = self.rnn(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # hidden = 0.5 * (hidden[self.n_layers - 1] + hidden[-1]) # Average over forward and backward hidden state at top layer
        hidden = torch.cat((hidden[self.n_layers - 1], hidden[-1]), 1) # Concatenate forward and backward hidden state at top layer
        if self.use_attention:
            hidden = self.attention(self.use_attention, outputs) # (batch, 2 * hidden_size)
        hidden = F.dropout(hidden, self.dropout[2], training=self.training) # Dropout for hidden state
        
        # Output
        if self.output_projection:
            hidden = self.linear(hidden)
            relu = self.relu(hidden)
            out = self.out(relu)
        else:
            out = self.out(hidden)
        return out.view(batch_size, self.output_size)
    
    def attention(self, attention_model, hidden_states):
        # Transform to size: (batch, seq_len, hidden_size * num_directions)
        hidden_states = hidden_states.view(hidden_states.size(1), hidden_states.size(0), hidden_states.size(2))
        if attention_model == "self-attention":
            nonlinear_states = F.tanh(hidden_states.matmul(self.attn_matrix.transpose(0, 1))) # (batch, seq_len, d)
            attn_weights = F.softmax(nonlinear_states.matmul(self.attn_vector)) # (batch, seq_len)
        elif attention_model == "vec-attention":
            nonlinear_states = F.tanh(hidden_states.matmul(self.attn_vector)) # (batch, seq_len)
            attn_weights = F.softmax(nonlinear_states) # (batch, seq_len)
        attn_weights = attn_weights.view(attn_weights.size(0), 1, attn_weights.size(1)) # (batch, 1, seq_len)
        weighted_hidden = attn_weights.bmm(hidden_states)# (batch, 1, seq_len), (batch, seq_len, 2*hidden_size) => (batch, 1, 2*hidden_size)
        return weighted_hidden


    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if self.model == 'lstm':
            return (Variable(weight.new(2 * self.n_layers, batch_size, self.hidden_size).zero_()),
                    Variable(weight.new(2 * self.n_layers, batch_size, self.hidden_size).zero_()))
        else:
            return Variable(weight.new(2 * self.n_layers, batch_size, self.hidden_size).zero_())