# -*- coding: utf-8 -*-
import os
import os.path as osp
import argparse
import numpy as np 
import json
import time
import random
import h5py

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_value_
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
# from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision import models

from inf_loader import inf_loader
from models import Similarity_Measure, Model_M, LanguageModelCriterion

from tqdm import tqdm 
# from misc.rewards import get_self_critical_reward, init_cider_scorer

def random_pick(some_list, probabilities): 
    x = random.uniform(0,1) 
    cumulative_probability = 0.0 
    for item, item_probability in zip(some_list, probabilities): 
        cumulative_probability += item_probability 
        if x < cumulative_probability:
            break
    return item

def find_glove(emb_dim, input_id, emb_npy, input_num, flag, input_num2=None):
  # if flag=='obj':
  if 1:
    out_emb = torch.FloatTensor(input_id.size(0),input_id.size(1),emb_dim).zero_()
    for bs_id,bs_rel in enumerate(input_id):
        # for word_id, word in enumerate(bs_rel):
        for word_id in range(input_num[bs_id]):
            word = input_id[bs_id,word_id]
            out_emb[bs_id,word_id] = torch.FloatTensor(emb_npy[word])
  
  return out_emb
    
def inference(args,split, model_M):
  batchsize = args.batchsize
  t_start = time.time()
  valtest_data = inf_loader(args, split=split)
  print('[DEBUG] Loading %s data ... %f secs' % (split, time.time() - t_start))

  valtest_data_loader = DataLoader(dataset=valtest_data, num_workers=1,batch_size=batchsize, shuffle=False, drop_last=True)
  
  model_M.train(False)
  
  nbatches = np.int_(np.floor((len(valtest_data.ids)*1.)/batchsize)) 
  
  obj_g_e = np.load(args.glove_emb_path+'obj_glove_emb.npy')
  rel_g_e = np.load(args.glove_emb_path+'pred_glove_emb.npy')
  scores=0.
  inf_out = []
  inffn = osp.join(args.model_dir, args.out_name)
  hfile = h5py.File(inffn,'w')
  
  for batch_idx, (img_feat, objs, real_rels, rel_mask, edges, rel_num, obj_num, img_id) in \
      tqdm(enumerate(valtest_data_loader), total=nbatches):
      
      img_feat = img_feat.cuda()
      obj_emb = find_glove(args.emb_dim,objs,obj_g_e, obj_num, flag='obj') 
      obj_bina_emb = torch.FloatTensor(img_feat.size(0),real_rels.size(1),2,args.emb_dim).zero_()
      obj_bina_emb_mean = torch.FloatTensor(real_rels.size(0),real_rels.size(1),args.emb_dim).zero_()
      obj_bina_out = torch.LongTensor(batchsize,real_rels.size(1),2)
      obj_bina_word = []
      for bs in range(edges.size(0)):
        obj_bina_word_tmp_l1 = []
        for rel_id, obj_bi in enumerate(edges[bs]):
          obj_bina_word_tmp_l2=[]
          for obj_id in range(2):
            obj_emb_tmp = obj_emb[bs,obj_bi[obj_id]]
            obj_bina_out[bs,rel_id,obj_id] = objs[bs,obj_bi[obj_id]]
            obj_bina_emb[bs,rel_id,obj_id] = obj_emb_tmp
          two_obj_emb = torch.cat([obj_bina_emb[bs,rel_id,0,:].unsqueeze(-1),obj_bina_emb[bs,rel_id,1,:].unsqueeze(-1)],dim=-1)
          two_obj_emb = two_obj_emb.mean(-1)
          obj_bina_emb_mean[bs,rel_id] = two_obj_emb
      obj_emb = obj_emb.cuda()
      obj_bina_emb = obj_bina_emb.cuda()
      obj_bina_emb_mean = obj_bina_emb_mean.cuda()
      out_rel_emb = model_M(img_feat, obj_emb, obj_bina_emb_mean)
      for i in range(batchsize):
        rel_save = torch.FloatTensor(rel_num[i],3,args.emb_dim)
        for rel_id in range(rel_num[i]):
            rel_selected = torch.FloatTensor(3,args.emb_dim)
            rel_selected[0]= obj_bina_out[i,rel_id,0]
            rel_selected[2]=obj_bina_out[i,rel_id,1]
            rel_selected[1]=out_rel_emb[i,rel_id]
            rel_save[rel_id]=rel_selected
        hfile.create_dataset(img_id[i], data=rel_save.cpu().float().detach().numpy())
      
      