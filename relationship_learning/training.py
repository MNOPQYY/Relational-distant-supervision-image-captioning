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
from models import Similarity_Measure, Model_M, LanguageModelCriterion, L2Criterion, rel_img_rec_trans

from tqdm import tqdm 
from val import valtest 
from inference import inference

torch.backends.cudnn.enabled=False

def find_glove(emb_dim, input_id, emb_npy, input_num, flag):
  out_emb = torch.FloatTensor(input_id.size(0),input_id.size(1),emb_dim).zero_()
  for bs_id,bs_rel in enumerate(input_id):
    for word_id in range(input_num[bs_id]):
        word = input_id[bs_id,word_id]
        out_emb[bs_id,word_id] = torch.FloatTensor(emb_npy[word])
  return out_emb

def train(args):
 
  t_start = time.time()
  train_data = coco_loader(args, split='train')
  print('[DEBUG] Loading train data ... %f secs' % (time.time() - t_start))
  
  train_data_loader = DataLoader(dataset=train_data, num_workers=1,batch_size=args.batchsize, shuffle=True, drop_last=True)
  
  sim_model = Similarity_Measure(args)
  sim_model.cuda()
  sim_model.train(True)
  
  model_M = Model_M(args, sim_model)
  model_M.cuda()
  model_M.train(True)
  
  model_rec = rel_img_rec_trans(args)
  model_rec.cuda()
  model_rec.train(True)
  
  m_optimizer = optim.Adam(model_M.parameters(), lr=args.learning_rate, weight_decay=0)
  r_optimizer = optim.Adam(model_rec.parameters(), lr=args.learning_rate, weight_decay=0)
  exp_lr_scheduler = lr_scheduler.StepLR(m_optimizer, step_size=args.lr_step_size, gamma=.8)
  r_exp_lr_scheduler = lr_scheduler.StepLR(r_optimizer, step_size=args.lr_step_size, gamma=.8)
  
  batchsize = args.batchsize
  nbatches = np.int_(np.floor((len(train_data.ids)*1.)/batchsize)) 
  bestscore = 10000000
  crit = L2Criterion()
  
  bestmodelfn = osp.join(args.model_dir, 'bestmodel.pth')
  if(osp.exists(bestmodelfn)):
    modelfn = torch.load(bestmodelfn)
    sim_model.load_state_dict(modelfn['sim_dict'])
    model_M.load_state_dict(modelfn['m_dict'])
    model_rec.load_state_dict(modelfn['r_dict'])
    model_M.calculate_sim = sim_model
    
  obj_g_e = np.load(args.glove_emb_path+'obj_glove_emb.npy')
  rel_g_e = np.load(args.glove_emb_path+'pred_glove_emb.npy')
  
  scores = valtest(args, split='val', model_M=model_M)
  for epoch in range(args.epochs):
    loss_train = 0.
    
    exp_lr_scheduler.step()
    r_exp_lr_scheduler.step()
    for batch_idx, (img_feat, objs, real_rels, rel_mask,rel_mask_none,rel_weight, edges, rel_num, obj_num, img_id, st_id) in \
      tqdm(enumerate(train_data_loader), total=nbatches):
      torch.cuda.synchronize()
      m_optimizer.zero_grad()
      r_optimizer.zero_grad()
      img_feat = img_feat.cuda()
      rel_mask = rel_mask.cuda()
      rel_mask_none = rel_mask_none.cuda()
      rel_weight = rel_weight.cuda()
      edges = edges.cuda()
      real_rels = real_rels.cuda()
      obj_emb = find_glove(args.emb_dim,objs,obj_g_e,obj_num,flag='obj')
      rela_emb = find_glove(args.emb_dim,real_rels,rel_g_e,rel_num,flag='rel')
      obj_bina_emb = torch.FloatTensor(real_rels.size(0),real_rels.size(1),2,args.emb_dim).zero_()
      obj_bina_emb_mean = torch.FloatTensor(real_rels.size(0),real_rels.size(1),args.emb_dim).zero_()
      for bs in range(edges.size(0)):
        for rel_id in range(rel_num[bs]):
          obj_bi = edges[bs,rel_id]
          for obj_id in range(2):
            obj_emb_tmp = obj_emb[bs, obj_bi[obj_id]]
            obj_bina_emb[bs,rel_id,obj_id] = obj_emb_tmp
          two_obj_emb = torch.cat([obj_bina_emb[bs,rel_id,0,:].unsqueeze(-1),obj_bina_emb[bs,rel_id,1,:].unsqueeze(-1)],dim=-1)
          two_obj_emb = two_obj_emb.mean(-1)
          obj_bina_emb_mean[bs,rel_id] = two_obj_emb
      obj_emb = obj_emb.cuda()
      rela_emb = rela_emb.cuda()
      obj_bina_emb = obj_bina_emb.cuda()
      obj_bina_emb_mean = obj_bina_emb_mean.cuda()
      out_rel_emb = model_M(img_feat, obj_emb, obj_bina_emb_mean)
      loss = crit(out_rel_emb, rela_emb,rel_mask,rel_weight)
      input_embedding = torch.cat([obj_bina_emb[:,:,0,:],obj_bina_emb[:,:,1,:]],dim=-1)
      input_embedding = torch.cat([input_embedding,out_rel_emb],dim=-1)
      loss_rec = model_rec(img_feat, input_embedding, rel_num)
      loss_train += loss.item()/100
      loss_train += loss_rec.item()*10
      loss.backward()
      if batch_idx%500==1:
        print('loss train in this batch:: %f, loss rec in this batch:: %f'%(loss.item()/100,loss_rec.item()*10))
      m_optimizer.step()
      r_optimizer.step()
      torch.cuda.synchronize()
    loss_train = (loss_train*1.)/(batch_idx)
    print("[DEBUG] epoch %d, train_loss = %.6f" %
                    (epoch, loss_train))
    modelfn = osp.join(args.model_dir, 'model.pth')
    torch.save({
        'epoch': epoch,
        'm_dict': model_M.state_dict(),
        'r_dict': model_rec.state_dict(),
        'sim_dict': sim_model.state_dict(),
        'optimizer' : m_optimizer.state_dict(),
        'optimizer_r' : r_optimizer.state_dict(),
      }, modelfn)
      
    if epoch<0:
        score=1000000000
    else:
        score = valtest(args, split='val', model_M = model_M)
        score = score/100
        print('val score:: %f ' % (score))        
    if score<bestscore:
        bestscore = score
        f = open(osp.join(args.model_dir, 'score_withnone_val.txt'),'a')
        f.write('loss_train:'+' '+str(loss_train)+'\n')
        f.write('val score:'+' '+str(score)+'\n')
        f.close()
        print('val score:: %f ' % (score))
        os.system('cp %s %s' % (modelfn, bestmodelfn))
    else:
        f = open(osp.join(args.model_dir, 'score_withnone_val.txt'),'a')
        f.write('loss_train:'+' '+str(loss_train)+'\n')
        f.write('val score:'+' '+str(score)+'\n')
        f.close()
