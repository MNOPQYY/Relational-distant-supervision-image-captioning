# -*- coding: utf-8 -*-
import os
import os.path as osp
import argparse
import numpy as np 
import json
import time
 

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_value_
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision import models

from coco_loader import coco_loader
from atts2vt import DecoderRNN, EncoderRNN, S2VTAttModel, EncoderPool
from atts2vt.TransformerDecoder import Transformer
from graph import GraphTripleConvNet
from atts2vt.ShareEmbedding import Share_Embedding

from tqdm import tqdm 
from valtest import valtest 
import misc.utils as utils
from misc.rewards import get_self_critical_reward, init_cider_scorer

def find_glove(emb_dim, input_id, emb_npy, input_num, flag, input_num2=None):
  out_emb = torch.FloatTensor(input_id.size(0),input_id.size(1),emb_dim).zero_()
  for bs_id,bs_rel in enumerate(input_id):
    for word_id in range(input_num[bs_id]):
        word = input_id[bs_id,word_id]
        out_emb[bs_id,word_id] = torch.FloatTensor(emb_npy[word])
  return out_emb

def orgnize_triple(obj_vecs, pred_vecs, edges, n_rel):
  
  bs, rel_num, gr_dim = pred_vecs.size()
  tri_feats = []
  for i in range(bs):
    obj_vec = obj_vecs[i]
    pred_vec = pred_vecs[i]
    edge = edges[i]
    h_idx = edge[..., 0]
    t_idx = edge[..., 1]
    cur_h_vec = obj_vec[h_idx]
    cur_t_vec = obj_vec[t_idx]
    tri_feat = torch.cat([cur_h_vec, pred_vec],1)
    tri_feat = torch.cat([tri_feat, cur_t_vec],1)
    tri_feat = tri_feat[:n_rel[i]]
    tri_feat = F.pad(tri_feat,(0,0,0,rel_num-n_rel[i].item()))
    tri_feats.append(tri_feat.unsqueeze(0))
  tri_feats = torch.cat(tri_feats,0)
  return tri_feats
  
def repeat_frm_per_cap(s2vt_feat, tri_feat, ncap_per_img):
  """Repeat image features ncap_per_img times"""
  
  batchsize, ds = s2vt_feat.size()
  _, ntri, dt = tri_feat.size()

  batchsize_cap = batchsize*ncap_per_img
  
  tri_feat = tri_feat.unsqueeze(1).expand(\
    batchsize, ncap_per_img, ntri, dt)

  tri_feat = tri_feat.contiguous().view(\
    batchsize_cap, ntri, dt)
    
  s2vt_feat = s2vt_feat.unsqueeze(1).expand(\
    batchsize, ncap_per_img, ds)

  s2vt_feat = s2vt_feat.contiguous().view(\
    batchsize_cap, ds)

  return s2vt_feat, tri_feat

def train(args):
  """Trains model for args.nepochs (default = 30)"""

  t_start = time.time()
  train_data = coco_loader(args, split='train', ncap_per_img=args.ncap_per_img)
  print('[DEBUG] Loading train data ... %f secs' % (time.time() - t_start))

  train_data_loader = DataLoader(dataset=train_data, num_workers=1,batch_size=args.batchsize, shuffle=True, drop_last=True)

  embed = Share_Embedding(args,train_data.numwords)
  embed.cuda()
  embed.train(True)
  
  
  model_gcn = GraphTripleConvNet(embed,
    300, # embeddim
    300, # inputdim
    512
    
    )
  model_gcn.cuda()
  model_gcn.train(True)
  
  
  model_trans = Transformer(train_data.wordlist, args.settings)
  model_trans.cuda()
  model_trans.train(True)
  
  gcn_optimizer = optim.RMSprop(model_gcn.parameters(), lr=args.learning_rate)
  optimizer = optim.RMSprop(model_trans.parameters(), lr=args.learning_rate)
  gcn_scheduler = lr_scheduler.StepLR(gcn_optimizer, step_size=args.lr_step_size, gamma=.8)
  exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=.8)  
  
  batchsize = args.batchsize
  batchsize_cap = batchsize
  max_tokens = train_data.max_tokens
  nbatches = np.int_(np.floor((len(train_data.ids)*1.)/batchsize)) 
  bestscore = .0
  crit = utils.LanguageModelCriterion()
  rl_crit = utils.RewardCriterion()
  kg_crit = utils.RotateKnowledge()
  
  bestmodelfn = osp.join(args.model_dir, 'bestmodel.pth')
  if(osp.exists(bestmodelfn)):
    modelfn = torch.load(bestmodelfn)
    embed.load_state_dict(modelfn['embed_dict'])
    model_trans.load_state_dict(modelfn['state_dict'])
    model_gcn.load_state_dict(modelfn['gcn_state_dict'])
    
    model_trans.decoder.embedding = embed
   
  obj_g_e = np.load(args.glove_emb_path+'obj_glove_emb.npy')
  rel_g_e = np.load(args.glove_emb_path+'pred_glove_emb.npy')
  
  for epoch in range(args.epochs):
    loss_train = 0.
    
    gcn_scheduler.step()
    exp_lr_scheduler.step()
    for batch_idx, (objs, rels, edges, rel_word_nums, n_rel, n_obj, captions, wordclass, mask, _) in \
      tqdm(enumerate(train_data_loader), total=nbatches):

      torch.cuda.synchronize()
      wordclass = wordclass.view(batchsize_cap, max_tokens)
      mask = mask.view(batchsize_cap, max_tokens)
      mask = mask.cuda().type(torch.cuda.FloatTensor)

      optimizer.zero_grad()
      gcn_optimizer.zero_grad()
      
      wordclass_v = wordclass.cuda()
      edges = edges.cuda().long()
      
      obj_emb = find_glove(args.emb_dim,objs,obj_g_e, n_obj, flag='obj') #all obj in img
      rela_emb = find_glove(args.emb_dim,rels,rel_g_e,n_rel,'rel')
      
      obj_emb = obj_emb.cuda()
      rela_emb = rela_emb.cuda()
      
      obj_feat, rel_feat = model_gcn(embed, objs,obj_emb, rels, rela_emb, edges, rel_word_nums)
      
      tri_feat = orgnize_triple(obj_feat, rel_feat, edges, n_rel) #bs, 6, d
      
      wordact = model_trans(tri_feat, wordclass_v, lengths=None)
      loss = crit(wordact, wordclass_v[:, 1:], mask[:, 1:])
                
      loss_train += loss.item()

      loss.backward()
      clip_grad_value_(model_trans.parameters(), 10.)
      clip_grad_value_(model_gcn.parameters(), 10.)
      gcn_optimizer.step()
      optimizer.step()
      torch.cuda.synchronize()
    loss_train = (loss_train*1.)/(batch_idx)
    
    print("[DEBUG] epoch %d, train_loss = %.6f" %
    
    modelfn = osp.join(args.model_dir, 'model.pth')
    torch.save({
        'epoch': epoch,
        'embed_dict': embed.state_dict(),
        'state_dict': model_trans.state_dict(),
        'gcn_state_dict': model_gcn.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'gcn_optimizer' : gcn_optimizer.state_dict(),
      }, modelfn)
      

    if epoch<0:
      score = 0
    else:
      scores = valtest(args, 'test', embed = embed, model_trans=model_trans, model_gcn=model_gcn) 
      score = scores[0][args.score_select]
    if(score > bestscore):
      bestscore = score
      f = open(osp.join(args.model_dir, 'score_test.txt'),'a')
      f.write('loss_train:'+' '+str(loss_train)+'\n')
      for k, v in scores[0].items():
        print('%s %f ' % (k, v))
        f.write(k+' '+str(v)+'\n')
      f.close()
      print('[DEBUG] Saving model at epoch %d with %s score of %f'\
        % (epoch, args.score_select, score))
      os.system('cp %s %s' % (modelfn, bestmodelfn))
      
      if(score >= 3.5):
        modelfn_save = args.model_dir+ '/bestmodel'+'{:.2f}'.format(score*100.)+'.pth'
        os.system('cp %s %s' % (modelfn, modelfn_save))
       