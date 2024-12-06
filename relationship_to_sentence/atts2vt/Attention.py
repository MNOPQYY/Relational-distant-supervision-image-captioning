import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Attention(nn.Module):
    """
    Applies an attention mechanism on the output features from the decoder.
    """

    def __init__(self, dim):
        super(Attention, self).__init__()
        self.dim = dim
        self.linear1 = nn.Linear(dim * 2, dim)
        self.linear2 = nn.Linear(dim, 1, bias=False)
        self._init_hidden()

    def _init_hidden(self):
        nn.init.kaiming_normal_(self.linear1.weight)
        nn.init.kaiming_normal_(self.linear2.weight)
        
    def _inplace_inf(self, e, n_rel):
        '''
        e -- batch_size * seq_len
        n_rel -- batch_size
        '''
        bs, sl = e.size()
        # print sl
        new_e = []
        for i in range(bs):
          e1 = e[i,:n_rel[i].item()]
          e1 = F.pad(e1,(0,sl-n_rel[i].item()),'constant',-np.inf)
          new_e.append(e1.unsqueeze(0))
        new_e = torch.cat(new_e,0)
        return new_e

    def forward(self, hidden_state, encoder_outputs, n_rel):
        """
        Arguments:
            hidden_state {Variable} -- batch_size x dim
            encoder_outputs {Variable} -- batch_size x seq_len x dim
            n_rel --batch_size,

        Returns:
            Variable -- context vector of size batch_size x dim
        """
        batch_size, seq_len, _ = encoder_outputs.size()
        # print encoder_outputs.size()
        hidden_state = hidden_state.unsqueeze(1).repeat(1, seq_len, 1)
        inputs = torch.cat((encoder_outputs, hidden_state),
                           2).view(-1, self.dim * 2)
        o = self.linear2(F.tanh(self.linear1(inputs)))
        e = o.view(batch_size, seq_len)
        e = self._inplace_inf(e, n_rel)
        alpha = F.softmax(e, dim=1) # batch_size * seq_len
        # print alpha
        # encoder_outputs[encoder_outputs==-np.inf]=0
        context = torch.bmm(alpha.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, alpha
