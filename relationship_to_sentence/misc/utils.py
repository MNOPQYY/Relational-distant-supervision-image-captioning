import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    # seq = seq.cpu()
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            # ix = seq[i, j].item()
            ix = seq[i, j]
            if ix > 0:
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix.item())]
            else:
                break
        out.append(txt)
    return out
    
def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


class RewardCriterion(nn.Module):

    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = to_contiguous(input).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = (seq > 0).float()
        
        # mask = torch.cat([mask.new(mask.size(0), 1).fill_(1).cuda(),
                         # mask[:, :-1]], 1).contiguous().view(-1)
        mask_ = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        # print mask
                         
        # print(reward.shape)
        # print(input.shape)
        # print(mask.shape)
        output_ = - input * reward * mask_
        output = torch.sum(output_) / torch.sum(mask_)

        return output


class LanguageModelCriterion(nn.Module):

    def __init__(self):
        super(LanguageModelCriterion, self).__init__()
        self.loss_fn = nn.NLLLoss(reduce=False)
        # self.loss_fn = nn.NLLLoss(reduce=False)

    def forward(self, logits, target, mask):
        """
        logits: shape of (N, seq_len, vocab_size)
        target: shape of (N, seq_len)
        mask: shape of (N, seq_len)
        """
        # truncate to the same size
        batch_size = logits.shape[0]
        target = target[:, :logits.shape[1]]
        mask = mask[:, :logits.shape[1]]
        logits = logits.contiguous().view(-1, logits.shape[2])
        target = target.contiguous().view(-1)
        mask = mask.contiguous().view(-1)
        # loss_word = F.nll_loss(wordact_t[maskids, ...], \
          # wordclass_t[maskids, ...].contiguous().view(maskids.shape[0]))
        # print logits.size()
        # print target.size()
        loss = self.loss_fn(logits, target)
        # output = torch.sum(loss * mask) / batch_size
        output = torch.sum(loss * mask) / torch.sum(mask)
        return output
        
class KnowledgeCriterion(nn.Module):

    def __init__(self):
        super(KnowledgeCriterion, self).__init__()
        self.criterion = nn.Softplus()
        # self.loss_fn = nn.NLLLoss(reduce=False)
        # self.loss_fn = nn.NLLLoss(reduce=False)

    def forward(self, tri_feat_org, alpha, mask):
        """
        tri_feat_org: shape of (bs, rel_num, 1536)
        alpha: shape of (bs, seq_len, rel_num)
        mask: shape of (bs, seq_len)
        """
        batch_size, rel_num, f_dim = tri_feat_org.size()
        _, seq_len = mask.size()
        # print tri_feat_org.size()
        # print alpha.size()
        # print mask.size()
        
        numtrue = torch.sum(mask)
        
        mask = torch.unsqueeze(mask,2)
        mask.expand_as(alpha)
        alpha = alpha-0.1
        alpha = alpha*mask
        alpha = alpha.unsqueeze(3).expand(batch_size, seq_len, rel_num, f_dim)
        tri_feat_org = tri_feat_org.unsqueeze(1).expand_as(alpha)
        tri_feat_org = tri_feat_org*alpha
        tri_feat_org.view(-1, f_dim)
        
        h_feat = tri_feat_org[...,:512]
        r_feat = tri_feat_org[...,512:1024]
        t_feat = tri_feat_org[...,1024:]

        h_re = h_feat[...,:256]
        h_im = h_feat[...,256:]
        r_re = r_feat[...,:256]
        r_im = r_feat[...,256:]
        t_re = t_feat[...,:256]
        t_im = t_feat[...,256:]
        
        score = -torch.sum(
                h_re * t_re * r_re
                + h_im * t_im * r_re
                + h_re * t_im * r_im
                - h_im * t_re * r_im,
                -1,
                )
        regul = (torch.mean(h_re ** 2)
                + torch.mean(h_im ** 2)
                + torch.mean(t_re ** 2)
                + torch.mean(t_im ** 2)
                + torch.mean(r_re ** 2)
                + torch.mean(r_im ** 2))
                
        output = torch.sum(self.criterion(score))/numtrue + 0.01 * regul


        return output

class RotateKnowledge(nn.Module):

    def __init__(self, gamma=6.0):
        super(RotateKnowledge, self).__init__()
        self.criterion = nn.Softplus()
        # self.loss_fn = nn.NLLLoss(reduce=False)
        # self.loss_fn = nn.NLLLoss(reduce=False)
        self.epsilon = 2.0
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        # self.embedding_range = nn.Parameter(
            # torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            # requires_grad=False
        # )

    def forward(self, tri_feat_org, alpha, mask):
        pi = 3.14159265358979323846
        batch_size, rel_num, f_dim = tri_feat_org.size()
        _, seq_len = mask.size()
        # print tri_feat_org.size()
        # print alpha.size()
        # print mask.size()
        
        numtrue = torch.sum(mask)
        '''
        mask = torch.unsqueeze(mask,2)
        mask.expand_as(alpha)
        alpha = alpha-0.1
        alpha = alpha*mask
        alpha = alpha.unsqueeze(3).expand(batch_size, seq_len, rel_num, f_dim)
        tri_feat_org = tri_feat_org.unsqueeze(1).expand_as(alpha)
        tri_feat_org = tri_feat_org*alpha
        tri_feat_org.view(-1, f_dim)
        '''
        h_feat = tri_feat_org[...,:512]
        r_feat = tri_feat_org[...,512:1024]
        t_feat = tri_feat_org[...,1024:]
        '''
        r_feat = tri_feat_org[...,512:768]
        t_feat = tri_feat_org[...,768:]
        '''
        h_re = h_feat[...,:256]
        h_im = h_feat[...,256:]
        '''
        phase_relation = relation/(self.embedding_range.item()/pi)
        r_re = torch.cos(phase_relation)
        r_im = torch.sin(phase_relation)
        '''
        r_re = r_feat[...,:256]
        r_im = r_feat[...,256:]
        t_re = t_feat[...,:256]
        t_im = t_feat[...,256:]
        
        re_score = h_re*r_re-h_im*r_im
        im_score = h_re*r_im+h_im+r_re
        
        re_score = re_score-t_re
        im_score = im_score-t_im
        
        score = torch.stack([re_score,im_score],dim=0)
        score = score.norm(dim = 0)
        # score =score.sum(dim = 2)
        # score = -torch.sum(score,-1,)
        score = torch.sum(score,dim=2)
        # print('origin score', score)
        
        # regul = (torch.mean(h_re ** 2)
                # + torch.mean(h_im ** 2)
                # + torch.mean(t_re ** 2)
                # + torch.mean(t_im ** 2)
                # + torch.mean(r_re ** 2)
                # + torch.mean(r_im ** 2))
                
        output = -torch.sum(F.logsigmoid(-score))/numtrue 
        # + 0.01 * regul


        return output
        
        # return score