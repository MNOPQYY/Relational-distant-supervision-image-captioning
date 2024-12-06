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
from graph import GraphTripleConvNet

from tqdm import tqdm 
import misc.utils as utils
from cocoeval import language_eval

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

def valtest(args, split, modelfn=None, embed=None, model_trans=None, model_gcn=None):
  """Trains model for args.nepochs (default = 30)"""
  batchsize = args.val_test_batchsize
  t_start = time.time()
  valtest_data = coco_loader(args, split=split, ncap_per_img=1)
  print('[DEBUG] Loading %s data ... %f secs' % (split, time.time() - t_start))

  valtest_data_loader = DataLoader(dataset=valtest_data, num_workers=1,batch_size=batchsize, shuffle=False, drop_last=True)

  embed = embed
  embed.train(False)
  
  model_gcn = model_gcn
  model_gcn.train(False)
  
  model_trans = model_trans
  model_trans.train(False)
  
  bs = batchsize
  max_tokens = valtest_data.max_tokens
  nbatches = np.int_(np.floor((len(valtest_data.ids)*1.)/batchsize)) 

  pred_captions = []
  results = []
  
  obj_g_e = np.load(args.glove_emb_path+'obj_glove_emb.npy')
  rel_g_e = np.load(args.glove_emb_path+'pred_glove_emb.npy')
  
  for batch_idx, (objs, rels, edges, rel_word_nums, n_rel, n_obj, _, _, _, img_ids) in \
      tqdm(enumerate(valtest_data_loader), total=nbatches):

      
    edges = edges.cuda().long()
    
    obj_emb = find_glove(args.emb_dim,objs,obj_g_e, n_obj, flag='obj') #all obj in img
    rela_emb = find_glove(args.emb_dim,rels,rel_g_e,n_rel,'rel')
    
    obj_emb = obj_emb.cuda()
    rela_emb = rela_emb.cuda()
      
    obj_feat, rel_feat = model_gcn(embed, objs,obj_emb, rels,rela_emb, edges, rel_word_nums)
    
    tri_feats = orgnize_triple(obj_feat, rel_feat, edges, n_rel) #bs, 6, d
      
     
    wordclass_feed = np.zeros((bs, max_tokens), dtype='int64')
    wordclass_feed[:,0] = valtest_data.wordlist.index('<S>') 

    for i, fn in enumerate(img_ids):
        tri_feat = tri_feats[i]
        with torch.no_grad():
                        rest, _ = model_trans.sample(tri_feat, beam_size=args.beam_size)
        results.append({'image_id': fn, 'caption': rest[0]})
  scores = language_eval(results, args.model_dir, split)


  model_gcn.train(True) 
  model_trans.train(True)
  embed.train(True)

  return scores 