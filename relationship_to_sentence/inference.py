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

from inf_loader import inf_loader
from atts2vt import DecoderRNN, EncoderRNN, S2VTAttModel, EncoderPool
from graph import GraphTripleConvNet
from atts2vt.ShareEmbedding import Share_Embedding

from atts2vt.TransformerDecoder import Transformer

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
  
  

def inference(args, split):
  """Trains model for args.nepochs (default = 30)"""
  batchsize = args.batchsize
  t_start = time.time()
  valtest_data = inf_loader(args, split=split, ncap_per_img=1)
  print('[DEBUG] Loading %s data ... %f secs' % (split, time.time() - t_start))

  valtest_data_loader = DataLoader(dataset=valtest_data, num_workers=1,batch_size=batchsize, shuffle=False, drop_last=True)
  
  embed = Share_Embedding(args,valtest_data.numwords)
  model_gcn = GraphTripleConvNet(embed,
    300,
    300,
    512
    )
  model_s2vt = Transformer(valtest_data.wordlist, args.settings)
  embed=embed.cuda()
  model_s2vt = model_s2vt.cuda()
  model_gcn = model_gcn.cuda()
  bestmodelfn = osp.join(args.model_dir, args.best_modelpth)
  modelfn = torch.load(bestmodelfn)
  embed.load_state_dict(modelfn['embed_dict'])
  model_s2vt.load_state_dict(modelfn['state_dict'])
  model_gcn.load_state_dict(modelfn['gcn_state_dict'])
    
  model_s2vt.decoder.embedding = embed
  bs = batchsize
  max_tokens = valtest_data.max_tokens
  nbatches = np.int_(np.floor((len(valtest_data.ids)*1.)/batchsize)) 

  pred_captions = []
  
  obj_g_e = np.load(args.glove_emb_path+'obj_glove_emb.npy')
  for batch_idx, (objs, rels, edges, rel_word_nums, n_rel, n_obj, img_ids) in \
      tqdm(enumerate(valtest_data_loader), total=nbatches):

      
    edges = edges.cuda().long()
    objs = objs.cuda().long()
    rels = rels.cuda()
    obj_emb = find_glove(args.emb_dim,objs,obj_g_e, n_obj, flag='obj') 
    rela_emb = rels
    obj_emb = obj_emb.cuda()
    obj_feat, rel_feat = model_gcn(embed, objs, obj_emb, rels, rela_emb, edges, rel_word_nums)
    tri_feat = orgnize_triple(obj_feat, rel_feat, edges, n_rel) 
    wordclass_feed = np.zeros((bs, max_tokens), dtype='int64')
    wordclass_feed[:,0] = valtest_data.wordlist.index('<S>') 
    outcaps = np.empty((bs, 0)).tolist()

    for j in range(bs):
      att_feat = tri_feat[j]
      rest, _ = model_s2vt.sample(att_feat, beam_size=1)
      pred_captions.append({'image_id': img_ids[j], 'caption': rest[0]})
      
    with open(args.out_name,'w') as output:
        json.dump(pred_captions,output)