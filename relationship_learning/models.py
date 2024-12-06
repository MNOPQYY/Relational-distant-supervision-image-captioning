import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.weight_norm import weight_norm
import math

def build_mlp(dim_list, activation='relu', batch_norm='batch',
              dropout=0, final_nonlinearity=False):
  layers = []
  for i in range(len(dim_list) - 1):
    dim_in, dim_out = dim_list[i], dim_list[i + 1]
    layers.append(nn.Linear(dim_in, dim_out))
    final_layer = (i == len(dim_list) - 2)
    if not final_layer:
      if batch_norm == 'batch':
        layers.append(nn.BatchNorm1d(dim_out))
      if activation == 'relu':
        layers.append(nn.ReLU())
      elif activation == 'leakyrelu':
        layers.append(nn.LeakyReLU())
  return nn.Sequential(*layers)
  
class L2Criterion(nn.Module):

    def __init__(self):
        super(L2Criterion, self).__init__()
        self.loss_fn = nn.MSELoss(reduce=False)

    def forward(self, target, pred_rel_emb ,mask,rel_weight):
        loss = self.loss_fn(target,pred_rel_emb)
        loss = loss.sum(-1)
        output = torch.sum(loss * rel_weight*mask)
        return output
    
class LanguageModelCriterion(nn.Module):

    def __init__(self):
        super(LanguageModelCriterion, self).__init__()
        self.loss_fn = nn.NLLLoss(reduce=False)

    def forward(self, logits, target, rel_candidate,rel_wordlist_num,target_num, rel_candidate_num,mask,rel_weight):
        """
        logits: shape of (N, seq_len, vocab_size)
        target: shape of (N, seq_len)
        mask: shape of (N, seq_len)
        """
        logits_full = torch.FloatTensor(rel_candidate.size(0),rel_candidate.size(1),rel_wordlist_num).zero_()
        for bs in range(rel_candidate.size(0)):
            for rel_id in range(target_num[bs]):
              for cand_id in range(rel_candidate_num[bs,rel_id]):
                cand_rel = rel_candidate[bs,rel_id,cand_id]
                cand_rel_prob = logits[bs,rel_id,cand_id]
                logits_full[bs,rel_id,cand_rel] = cand_rel_prob
        logits_full = logits_full.cuda()
        batch_size = logits.shape[0]
        target = target[:, :logits_full.shape[1]]
        logits_full = logits_full.contiguous().view(-1, logits_full.shape[2])
        target = target.contiguous().view(-1)
        mask = mask.contiguous().view(-1)
        rel_weight = rel_weight.contiguous().view(-1)
        loss = self.loss_fn(logits_full, target)
        output = torch.sum(loss * rel_weight) / torch.sum(mask)
        return output

class Share_Embedding(nn.Module):
    def __init__(self, opt, vocab_size):
        super(Share_Embedding, self).__init__()
        
        self.embed_dim = opt.embed_dim
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.embedding.weight.data.uniform_(-0.1, 0.1)
    def forward(self, input_sequence, sinlge_word=True, rel_word_num=None):
        result = self.embedding(input_sequence)
        if not sinlge_word:
            result_rel = torch.FloatTensor(input_sequence.size(0),input_sequence.size(1),input_sequence.size(2),self.embed_dim)
            for bs_id, rel_bs in enumerate(result):
                for rel_id, rel in enumerate(rel_bs):
                    for cand_id, cand_rel in enumerate(rel):
                        result_rel[bs_id,rel_id,cand_id] = torch.mean(rel[:rel_word_num[bs_id,rel_id],cand_id])
            result = result_rel.cuda()
        return result

class Similarity_Measure(nn.Module):
    def __init__(self, opt):
        super(Similarity_Measure, self).__init__()
        self.img_dim = opt.feat_dim+2*opt.emb_dim
        self.rel_dim = opt.emb_dim
        self.hidden_dim = opt.hidden_dim
        fc_layers = [self.img_dim,self.hidden_dim,self.rel_dim]
        self.fc_sim = build_mlp(fc_layers,batch_norm= 'None')
    def forward(self, input_img):
        output = self.fc_sim(input_img)
        return output

class Model_M(nn.Module):
    def __init__(self, opt, sim_measure_model):
        super(Model_M, self).__init__()
        
        self.feat_dim = opt.feat_dim
        self.hidden_dim = opt.hidden_dim
        self.emb_dim = opt.emb_dim
        
        self.calculate_sim = sim_measure_model
        
    def _inplace_inf(self, e, n_rel):
        '''
        e -- batch_size * seq_len
        n_rel -- batch_size
        '''
        bs, sl, sd = e.size()
        new_e = torch.FloatTensor(e.size(0),e.size(1),e.size(2)).zero_()
        for i in range(bs):
         for j in range(sl):
          
          if n_rel[i,j]!=0:
            e1 = e[i,j,:n_rel[i,j].item()]
            e1 = F.pad(e1,(0,sd-n_rel[i,j].item()),'constant',-np.inf)
          else:
            e1 = e[i,j,:]
          new_e[i,j,:]=e1
        return new_e
    
    def forward(self, img_feat, obj_all, obj_bina_emb):
        obj_all = obj_all.mean(1) 
        im_obj = torch.cat([img_feat,obj_all],dim=1)
        im_obj = im_obj.unsqueeze(1)
        im_obj = im_obj.expand(im_obj.size(0),obj_bina_emb.size(1),im_obj.size(2))
        input_img = torch.cat([im_obj,obj_bina_emb],dim=-1)
        out_rel_emb = sim_score = self.calculate_sim(input_img)
        return out_rel_emb
        
class rel_img_rec(nn.Module):
    def __init__(self, opt, num_layers=1):
        super(rel_img_rec, self).__init__()

        self.decoder_dim = opt.decoder_dim
        self.output_dim = opt.feat_dim
        self.dropout = opt.dropout
        self.embed_dim = 3*opt.emb_dim
        
        self.language_model = nn.LSTM(self.embed_dim, self.decoder_dim, num_layers, batch_first=True)
        self.fc = weight_norm(nn.Linear(self.decoder_dim, self.output_dim))
        self.rec_loss = nn.MSELoss()
        self.init_weights()
    def init_weights(self):
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)
    
    def forward(self,img_feat, input_embedding, input_len, state=None):
        bs = input_embedding.size(0)
        language_out,(hiddens, cells) = self.language_model(input_embedding, state)
        out = self.fc(language_out)
        pred = torch.FloatTensor(out.size(0),out.size(-1)).cuda()
        for bs_idx in range(bs):
            pred[bs_idx]=out[bs_idx,max(0,input_len[bs_idx]-1),:]
            
        loss = self.rec_loss(img_feat,pred)
        return loss
        
class MultiHeadAttention(nn.Module):
    def __init__(self, settings):
        super(MultiHeadAttention, self).__init__()
        assert settings['d_model'] % settings['h'] == 0
        self.h = settings['h']
        self.d_k = settings['d_model'] // settings['h']
        self.linears = nn.ModuleList([nn.Linear(settings['d_model'], settings['d_model']) for _ in range(4)])
        self.drop = nn.Dropout(settings['dropout_p'])

    def _attention(self, query, key, value, mask=None):
        scores = query.matmul(key.transpose(-2, -1)) \
                 / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        scores = scores.softmax(-1)
        return scores.matmul(value)

    def forward(self, query, key, value, mask=None):
        """
            query: bs*n1*d_model
            key: bs*n2*d_model
            value: bs*n2*d_model
            mask: bs*(n2 or 1)*n2
        """
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)

        query, key, value = \
            [self.drop(l(x)).reshape(batch_size, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears[:3], (query, key, value))]  
        x = self._attention(query, key, value, mask)  
        x = x.transpose(1, 2).reshape(batch_size, -1, self.h * self.d_k)
        return self.drop(self.linears[-1](x))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, settings):
        super(PositionwiseFeedForward, self).__init__()
        self.pff = nn.Sequential(
            nn.Linear(settings['d_model'], settings['d_ff']),
            nn.ReLU(),
            nn.Linear(settings['d_ff'], settings['d_model'])
        )

    def forward(self, x):
        return self.pff(x)


class PositionalEncoding(nn.Module):
    def __init__(self, settings):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(settings['max_seq_len'], settings['d_model'])
        position = torch.arange(0, settings['max_seq_len']).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, settings['d_model'], 2).float() *
                             -(math.log(10000.0) / settings['d_model']))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.drop = nn.Dropout(settings['dropout_p'])

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.drop(x)


class EncoderLayer(nn.Module):
    def __init__(self, settings):
        super(EncoderLayer, self).__init__()
        self.multi_head_att = MultiHeadAttention(settings)
        self.feed_forward = PositionwiseFeedForward(settings)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(settings['d_model']) for _ in range(2)])
        self.drop = nn.Dropout(settings['dropout_p'])

    def _add_res_connection(self, x, sublayer, n):
        return x + self.drop(sublayer(self.layer_norms[n](x)))

    def forward(self, x, mask):
        x = self._add_res_connection(x, lambda x: self.multi_head_att(x, x, x, mask), 0)
        return self._add_res_connection(x, self.feed_forward, 1)


class Encoder(nn.Module):
    def __init__(self, settings):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(settings) for _ in range(settings['N_enc'])])
        self.layer_norm = nn.LayerNorm(settings['d_model'])

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.layer_norm(x)

class rel_img_rec_trans(nn.Module):
    def __init__(self, opt):
        super(rel_img_rec_trans, self).__init__()

        self.d_model = opt.decoder_dim
        self.output_dim = opt.feat_dim
        self.dropout = opt.dropout
        self.embed_dim = 3*opt.emb_dim
        self.att_embed = nn.Sequential(nn.Linear(self.embed_dim, self.d_model),
                                       nn.ReLU())
        self.recons_token=nn.Parameter(torch.zeros(1,1,self.embed_dim))
        self.encoder = Encoder(opt.settings)
        self.fc = nn.Linear(self.d_model, self.output_dim)
        self.rec_loss = nn.MSELoss()

    def forward(self, img_feat, input_embedding, input_len, att_masks=None):
        
        bs = input_embedding.size(0)
        recons_tokens=self.recons_token.expand(bs,1,input_embedding.size(-1))
        input_embedding=torch.cat((recons_tokens,input_embedding),dim=1)
        att_feats, att_masks = self._feats_encode(input_embedding, att_masks)
        enc_out = self.encoder(att_feats, att_masks)
        recon_feat = self.fc(enc_out[:, 0])
        loss = self.rec_loss(img_feat,recon_feat)
        return loss
        
    def _feats_encode(self, att_feats, att_masks=None):
        att_feats = att_feats.reshape(att_feats.size(0), -1, att_feats.size(-1))
        att_feats = self.att_embed(att_feats)

        if att_masks is not None:
            att_masks = att_masks.unsqueeze(-2)
            
        return att_feats, att_masks