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
from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision import models

from coco_loader import coco_loader
from models import Similarity_Measure, Model_M, LanguageModelCriterion, L2Criterion

from tqdm import tqdm 

def calculate_accuracy(sim_score,real_rel,real_cand_id,rel_word_num,mask,mask_none):
    pred_result5_prob, pred_result5_index = torch.topk(sim_score,1,dim=-1)
    pred_result5 = torch.gather(real_cand_id,-1,pred_result5_index)
    real_rel = real_rel.unsqueeze(-1)
    real_rel_exp = real_rel.expand(pred_result5.size(0),pred_result5.size(1),pred_result5.size(2))
    com = pred_result5.eq(real_rel_exp).float()
    com = torch.where(pred_result5_prob==-1,torch.FloatTensor(1).zero_().cuda(),com)
    com = com.sum(-1)
    accuracy=torch.sum(com*mask)/torch.sum(mask)
    accuracy_without_none=torch.sum(com*mask_none)/torch.sum(mask_none)
    return accuracy,accuracy_without_none

def find_glove(emb_dim, input_id, emb_npy, input_num, flag):
  
  out_emb = torch.FloatTensor(input_id.size(0),input_id.size(1),emb_dim).zero_()
  for bs_id,bs_rel in enumerate(input_id):
    for word_id in range(input_num[bs_id]):
        word = input_id[bs_id,word_id]
        out_emb[bs_id,word_id] = torch.FloatTensor(emb_npy[word])
  return out_emb
    
def valtest(args,split, model_M):
  batchsize = 100
  t_start = time.time()
  valtest_data = coco_loader(args, split=split, ncap_per_img=1)
  print('[DEBUG] Loading %s data ... %f secs' % (split, time.time() - t_start))

  valtest_data_loader = DataLoader(dataset=valtest_data, num_workers=1,batch_size=batchsize, shuffle=False, drop_last=False)
  
  model_M.train(False)
  
  nbatches = np.int_(np.floor((len(valtest_data.ids)*1.)/batchsize)) 
  
  obj_g_e = np.load(args.glove_emb_path+'obj_glove_emb.npy')
  rel_g_e = np.load(args.glove_emb_path+'pred_glove_emb.npy')
  scores=0.
  score_wihtout_nones=0.
  val_out = []
  crit = L2Criterion()
  
  for batch_idx, (img_feat, objs, real_rels, rel_mask,rel_mask_none,rel_weight, edges, rel_num, obj_num, img_id,st_id) in \
      tqdm(enumerate(valtest_data_loader), total=nbatches):
      
      img_feat = img_feat.cuda()
      real_rels = real_rels.cuda()
      rel_mask = rel_mask.cuda()
      rel_mask_none=rel_mask_none.cuda()
      rel_weight = rel_weight.cuda()
      edges = edges.cuda()
      obj_emb = find_glove(args.emb_dim,objs,obj_g_e,obj_num,flag='obj')
      rela_emb = find_glove(args.emb_dim,real_rels,rel_g_e,rel_num,flag='rel')
      obj_bina_emb = torch.FloatTensor(img_feat.size(0),real_rels.size(1),2,args.emb_dim).zero_()
      obj_bina_emb_mean = torch.FloatTensor(real_rels.size(0),real_rels.size(1),args.emb_dim).zero_()
      obj_bina_out = torch.LongTensor(batchsize,real_rels.size(1),2)
      obj_bina_word = []
      for bs in range(edges.size(0)):
        obj_bina_word_tmp_l1 = []
        for rel_id, obj_bi in enumerate(edges[bs]):
          for obj_id in range(2):
            obj_emb_tmp = obj_emb[bs,obj_bi[obj_id]]
            obj_bina_emb[bs,rel_id,obj_id] = obj_emb_tmp
          two_obj_emb = torch.cat([obj_bina_emb[bs,rel_id,0,:].unsqueeze(-1),obj_bina_emb[bs,rel_id,1,:].unsqueeze(-1)],dim=-1)
          two_obj_emb = two_obj_emb.mean(-1)
          obj_bina_emb_mean[bs,rel_id] = two_obj_emb
      obj_emb = obj_emb.cuda()
      rela_emb = rela_emb.cuda()
      obj_bina_emb = obj_bina_emb.cuda()
      obj_bina_emb_mean = obj_bina_emb_mean.cuda()
      out_rel_emb = model_M(img_feat, obj_emb, obj_bina_emb_mean)
      score = crit(out_rel_emb, rela_emb,rel_mask,rel_weight)
      scores+=score.item()
  scores = scores/(nbatches*1.0)
  model_M.train(True)
  return scores
      