import torch.nn as nn
import torch


class EncoderPool(nn.Module):
    def __init__(self, dim_vid, dim_hidden, input_dropout_p=0.2):
        """

        Args:
            hidden_dim (int): dim of hidden state of rnn
            input_dropout_p (int): dropout probability for the input sequence
            dropout_p (float): dropout probability for the output sequence
            n_layers (int): number of rnn layers
            rnn_cell (str): type of RNN cell ('LSTM'/'GRU')
        """
        super(EncoderPool, self).__init__()
        self.dim_vid = dim_vid
        self.dim_hidden = dim_hidden
        self.input_dropout_p = input_dropout_p

        self.vid2hid = nn.Linear(dim_vid, dim_hidden)
        self.input_dropout = nn.Dropout(input_dropout_p)
        self.actv = nn.LeakyReLU()
        self.norm = nn.BatchNorm1d(dim_hidden)

        self._init_hidden()

    def _init_hidden(self):
        nn.init.kaiming_normal_(self.vid2hid.weight)

    def forward(self, tri_feat, vid_feats, mode='train'):
        """
        Applies a multi-layer RNN to an input sequence.
        Args:
            input_var (batch, seq_len): tensor containing the features of the input sequence.
            input_lengths (list of int, optional): A list that contains the lengths of sequences
              in the mini-batch
        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
        """
        
        
        batch_size, seq_len, dim_vid = tri_feat.size()
        
        vid_feats = self.vid2hid(vid_feats)
        vid_feats = self.norm(vid_feats)
        vid_feats = self.actv(vid_feats)
        if mode == 'train':
            vid_feats = self.input_dropout(vid_feats)
        
        # output =  tri_feat
        hidden = torch.zeros(2, batch_size, self.dim_hidden).cuda()
        cell_state = torch.zeros(2, batch_size, self.dim_hidden).cuda()
        # print hidden.size()
        return tri_feat, hidden, cell_state, vid_feats
