import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from .Attention import Attention


class DecoderRNN(nn.Module):
    """
    Provides functionality for decoding in a seq2seq framework, with an option for attention.
    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        dim_hidden (int): the number of features in the hidden state `h`
        n_layers (int, optional): number of recurrent layers (default: 1)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        bidirectional (bool, optional): if the encoder is bidirectional (default False)
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        rnn_dropout_p (float, optional): dropout probability for the output sequence (default: 0)

    """

    def __init__(self,
                 vocab_size,
                 max_len,
                 dim_hidden,
                 dim_word,
                 n_layers=2,
                 rnn_cell='gru',
                 bidirectional=False,
                 input_dropout_p=0.2,
                 rnn_dropout_p=0.1):
        super(DecoderRNN, self).__init__()

        self.bidirectional_encoder = bidirectional

        self.dim_output = vocab_size
        self.dim_hidden = dim_hidden * 2 if bidirectional else dim_hidden
        self.dim_word = dim_word
        self.max_length = max_len
        self.cell_name = rnn_cell
        # self.sos_id = 1
        # self.eos_id = 0
        self.input_dropout = nn.Dropout(input_dropout_p)
        self.embedding = nn.Embedding(self.dim_output, dim_word)
        self.attention = Attention(self.dim_hidden)
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTMCell
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRUCell
        # self.rnn = nn.GRU(
            # self.dim_hidden + dim_word,
            # self.dim_hidden,
            # n_layers,
            # batch_first=True,
            # dropout=rnn_dropout_p)
        self.rnn_TDA = self.rnn_cell(
            2*self.dim_hidden + dim_word,
            self.dim_hidden)
        # self.rnn_lg = self.rnn_cell(
            self.dim_hidden*2,
            self.dim_hidden)
        self.out = nn.Linear(self.dim_hidden, self.dim_output)
        self.triin = nn.Linear(self.dim_hidden*3, self.dim_hidden)

        self._init_weights()

    def forward(self,
                vid_feats,
                encoder_outputs,
                encoder_hidden,
                encoder_cell,
                n_rel,
                targets=None,
                mode='train',
                temperature=1.,
                sample_max = True):
        """

        Inputs: inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
        - **encoder_hidden** (num_layers * num_directions, batch_size, dim_hidden): tensor containing the features in the
          hidden state `h` of encoder. Used as the initial hidden state of the decoder. (default `None`)
        - **encoder_outputs** (batch, seq_len, dim_hidden * num_directions): (default is `None`).
        - **targets** (batch, max_length): targets labels of the ground truth sentences

        Outputs: seq_probs,
        - **seq_logprobs** (batch_size, max_length, vocab_size): tensors containing the outputs of the decoding function.
        - **seq_preds** (batch_size, max_length): predicted symbols
        """
        
        # print sample_max
        # sample_max = 1
        # beam_size = 1
        # temperature = opt.get('temperature', 1.0)
        # temperature = 1.

        batch_size, _, feat_size = encoder_outputs.size()
        decoder_hidden = self._init_rnn_state(encoder_hidden)
        # print decoder_hidden.size()
        if self.cell_name.lower() == 'lstm':
          decoder_cell = self._init_rnn_state(encoder_cell)
          # print decoder_cell.size()
          decoder_cell_TDA = decoder_cell[0]
          # decoder_cell_lg = decoder_cell[1]
        # decoder_cell = self._init_rnn_state(encoder_cell)
        
        decoder_hidden_TDA = decoder_hidden[0]
        
        # decoder_hidden_lg = decoder_hidden[1]
        
        # print encoder_hidden.size()
        # print decoder_hidden.size()
        # print encoder_outputs.size()

        seq_logprobs = []
        seq_preds = []
        # self.rnn_TDA.flatten_parameters()
        # self.rnn_lg.flatten_parameters()
        # print targets[:, 0]
        # print targets_emb.is_contiguous()
        # print targets
        
        # encoder_outputs_avg = torch.mean(encoder_outputs,dim=1)
        encoder_outputs_avg = vid_feats
        encoder_outputs = self.triin(encoder_outputs)
        encoder_outputs = F.tanh(encoder_outputs)
        
        # if mode == 'train' or beam_size > 1:
        if mode == 'ttt':
          # seq_logprobs = decoder_hidden[0,:,:14]
          seq_logprobs = encoder_hidden[0,:,:14]
          seq_preds = encoder_outputs[:,0,:14]
        elif mode == 'inference':
            # print 'haha'
            current_words = self.embedding((torch.ones(batch_size)*102).cuda().long())
            # current_words = self.embedding((np.ones((batch_size))*102).cuda().long())
            # print current_words[0,:]
            # current_words = targets_emb[:, 0, :]
            # print current_words[0,:]
            alpha_all = []
            for t in range(self.max_length - 1):
                # encoder_outputs_avg = torch.mean(encoder_outputs,dim=1)
                decoder_input = torch.cat([decoder_hidden_lg, encoder_outputs_avg], dim=1)
                decoder_input = torch.cat([decoder_input, current_words], dim=1)
                
                if self.cell_name.lower() == 'lstm':
                  decoder_hidden_TDA, decoder_cell_TDA = self.rnn_TDA(
                      decoder_input, (decoder_hidden_TDA, decoder_cell_TDA))
                elif self.cell_name.lower() == 'gru':
                  decoder_hidden_TDA = self.rnn_TDA(decoder_input, decoder_hidden_TDA)
                context, alpha = self.attention(decoder_hidden_TDA, encoder_outputs, n_rel)
                alpha_all.append(alpha.view(batch_size,1,-1))
                decoderlg_input = torch.cat([context, decoder_hidden_TDA], dim=1)
                if self.cell_name.lower() == 'lstm':
                  decoder_hidden_lg, decoder_cell_lg = self.rnn_lg(
                      decoderlg_input, (decoder_hidden_lg, decoder_cell_lg))
                elif self.cell_name.lower() == 'gru':
                  decoder_hidden_lg = self.rnn_lg(
                      decoderlg_input, decoder_hidden_lg)
                decoder_output = decoder_hidden_lg
                logprobs = F.log_softmax(
                    self.out(decoder_output), dim=1)
                
                if sample_max:
                  #print 'hahasds'
                  sampleLogprobs, it = torch.max(logprobs, 1)
                  seq_logprobs.append(sampleLogprobs.view(-1, 1))
                  it = it.view(-1).long()
                else:
                  # print 'lala'

                  if temperature == 1.0:
                    prob_prev = torch.exp(logprobs)
                  else:
                    prob_prev = torch.exp(torch.div(logprobs, temperature))
                
                  it = torch.multinomial(prob_prev, 1).cuda()
                  sampleLogprobs = logprobs.gather(1, it)
                  seq_logprobs.append(sampleLogprobs.view(-1, 1))
                  it = it.view(-1).long()
                

                seq_preds.append(it.view(-1, 1))
                current_words = self.embedding(it)

            seq_logprobs = torch.cat(seq_logprobs, 1)
            seq_preds = torch.cat(seq_preds, 1)
            alpha_all = torch.cat(alpha_all, 1)
            
            for i in range(batch_size):
              if not torch.sum(seq_preds[i]==0)==0:
                num_words = torch.nonzero(seq_preds[i]==0)[0][0]
                seq_preds[i,num_words:].fill_(0)
                seq_logprobs[i,num_words:].fill_(0)
                


        else:
            targets_emb = self.embedding(targets)
            # print 'lala'
            # if beam_size > 1:
                # return self.sample_beam(encoder_outputs, decoder_hidden, opt)

            # use targets as rnn inputs
            # targets_emb = self.embedding(targets)
            alpha_all = []
            for i in range(self.max_length-1):
                current_words = targets_emb[:, i, :]
                # print current_words.size()
                # encoder_outputs_avg = torch.mean(encoder_outputs,dim=1)
                # print encoder_outputs_avg.size()
                decoder_input = torch.cat([decoder_hidden_lg, encoder_outputs_avg], dim=1)
                # print decoder_input.size()
                # print current_words.size()
                decoder_input = torch.cat([decoder_input, current_words], dim=1)
                if mode == 'train':
                    decoder_input = self.input_dropout(decoder_input)
                    encoder_outputs = self.input_dropout(encoder_outputs)
                # print decoder_input.size()
                if self.cell_name.lower() == 'lstm':
                  decoder_hidden_TDA, decoder_cell_TDA = self.rnn_TDA(
                      decoder_input, (decoder_hidden_TDA, decoder_cell_TDA))
                elif self.cell_name.lower() == 'gru':
                  # print decoder_input.size()
                  # print decoder_hidden_TDA.size()
                  decoder_hidden_TDA = self.rnn_TDA(
                      decoder_input, decoder_hidden_TDA)
                # print "haha"
                # print decoder_hidden_TDA.size()
                # context, alpha = self.attention(decoder_hidden_TDA, encoder_outputs, n_rel)
                # alpha_all.append(alpha.view(batch_size,1,-1))
                # decoderlg_input = torch.cat([context, decoder_hidden_TDA], dim=1)
                if self.cell_name.lower() == 'lstm':
                  # decoder_hidden_lg, decoder_cell_lg = self.rnn_lg(
                      # decoderlg_input, (decoder_hidden_lg, decoder_cell_lg))
                # elif self.cell_name.lower() == 'gru':
                  # decoder_hidden_lg = self.rnn_lg(
                      # decoderlg_input, decoder_hidden_lg)
                # decoder_output = decoder_hidden_lg
                logprobs = F.log_softmax(
                    self.out(decoder_hidden_TDA), dim=1)
                seq_logprobs.append(logprobs.unsqueeze(1))
            '''
            for i in range(self.max_length - 1):
                current_words = targets_emb[:, i, :]
                context = self.attention(decoder_hidden.squeeze(0), encoder_outputs)
                # bottom up and top down!!!!!
                decoder_input = torch.cat([current_words, context], dim=1)
                if mode == 'train':
                    decoder_input = self.input_dropout(decoder_input)
                decoder_input = decoder_input.unsqueeze(1)
                decoder_output, decoder_hidden = self.rnn(
                    decoder_input, decoder_hidden)
                logprobs = F.log_softmax(
                    self.out(decoder_output.squeeze(1)), dim=1)
                seq_logprobs.append(logprobs.unsqueeze(1))

            
            '''
            seq_logprobs = torch.cat(seq_logprobs, 1)
            # alpha_all = torch.cat(alpha_all, 1)
            # print seq_logprobs.size()
            

        return seq_logprobs, seq_preds
        # , alpha_all

    def _init_weights(self):
        """ init the weight of some layers
        """
        nn.init.kaiming_normal_(self.out.weight)
        nn.init.kaiming_normal_(self.triin.weight)

    def _init_rnn_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple(
                [self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, dim_hidden) -> (#layers, #batch, #directions * dim_hidden)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h
