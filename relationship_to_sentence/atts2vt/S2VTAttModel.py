import torch.nn as nn
import torch

class S2VTAttModel(nn.Module):
    def __init__(self, decoder, dim_hidden):
        """

        Args:
            encoder (nn.Module): Encoder rnn
            decoder (nn.Module): Decoder rnn
        """
        super(S2VTAttModel, self).__init__()
        # self.encoder = encoder
        self.decoder = decoder
        self.dim_hidden = dim_hidden

    def forward(self, tri_feat, n_rel, target_variable=None,
                mode='train', sample_max = True):
        """

        Args:
            vid_feats (Variable): video feats of shape [batch_size, seq_len, dim_vid]
            target_variable (None, optional): groung truth labels

        Returns:
            seq_prob: Variable of shape [batch_size, max_len-1, vocab_size]
            seq_preds: [] or Variable of shape [batch_size, max_len-1]
            alpha: [batch_size, max_len-1, triplet_number]
        """
        # encoder_outputs, encoder_hidden, encoder_cell, vid_feats = self.encoder(tri_feat, vid_feats, mode)
        batch_size = tri_feat.size(0)
        encoder_hidden = torch.zeros(2, batch_size, self.dim_hidden).cuda()
        encoder_cell = torch.zeros(2, batch_size, self.dim_hidden).cuda()
        seq_prob, seq_preds, alpha = self.decoder(tri_feat, encoder_hidden, encoder_cell, n_rel, target_variable, mode, sample_max=sample_max)
        return seq_prob, seq_preds, alpha
