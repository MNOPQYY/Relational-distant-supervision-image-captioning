# -*- coding: utf-8 -*-
from __future__ import print_function

import glob
import math
import numpy as np
import os
import os.path as osp
import string 
import pickle
import json
import nltk
import sys
import array
import random
import h5py
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
class coco_loader(Dataset):
  """Loads train/val/test splits of coco dataset"""

  def __init__(self, args, split='train'):
    self.split = split
    self.max_rel_num = args.max_rel_num
    self.max_rel_word_num = args.max_rel_word_num
    self.max_obj_num = args.max_obj
    
    self.f_fc = h5py.File(args.img_feature_path,'r')
    
    anno_name = args.sent_align_path+self.split+'.json'
    print('Loading annotation file...')
    with open(anno_name,'r') as input:
        anno_file = json.load(input)
    anno_file=anno_file['images']
    annos = {}
    for img_id, img in enumerate(anno_file):
            annos[img['filename']] = img
    self.annos = annos
    self.ids = list(self.annos.keys())
    
    print('Found %d images in split: %s'%(len(self.ids), self.split))
    
    obj_dict_tmp = json.load(open('./preprocess/gcc_obj_list.json', 'r'))
    self.obj_list = obj_dict_tmp
    self.numwords_obj = len(self.obj_list)
    print('[DEBUG] #words in object wordlist: %d' % (self.numwords_obj))
    
    rel_dict_tmp = json.load(open('./preprocess/gcc_pred_list.json', 'r'))
    self.rel_list = rel_dict_tmp
    self.numwords_rel = len(self.rel_list)
    print('[DEBUG] #words in relation wordlist: %d' % (self.numwords_rel))
    self.objdetlist = json.load(open(args.obj_list_path,'r'))

  def __getitem__(self, idx):    
    st_id = self.ids[idx]
    anno = self.annos[st_id]
    img_id = anno['filename']
    relations = anno['relation']
    obj_img_result = anno['objs_from_sent']
    obj_img_result = list(set(obj_img_result).intersection(set(self.objdetlist)))

    rel_mask = torch.FloatTensor(self.max_rel_num).zero_()
    rel_mask_none = torch.FloatTensor(self.max_rel_num).zero_()
    rel_weight = torch.FloatTensor(self.max_rel_num).zero_()
    rels = torch.LongTensor(self.max_rel_num).zero_()
    edges = torch.LongTensor(self.max_rel_num, 2).zero_()
    objs = torch.LongTensor(self.max_obj_num).zero_()
    bound = min(len(obj_img_result), self.max_obj_num)
    for ii in range(bound):
      word = obj_img_result[ii]
      if word not in self.obj_list:
        word = 'unknown'
      objs[ii] = self.obj_list.index(word)
    rel_dict = {}
    for rel_id,rel in enumerate(relations):
      if rel_id<self.max_rel_num:
        sub=rel[0]
        if sub not in self.obj_list:
            sub='unknown'
        obj=rel[2]
        if obj not in self.obj_list:
            obj='unknown'
        subid,objid = self.obj_list.index(sub),self.obj_list.index(obj)
        relation = rel[1]
        if relation not in self.rel_list:
            relation = 'unknown'
        rel_dict['_'.join(sorted([str(subid),str(objid)]))] = self.rel_list.index(relation)
    rel_id = 0
    for ii in range(bound):
        for jj in range(ii,bound):
            if ii == jj:
                continue
            pair_key = '_'.join(sorted([str(objs[ii].item()),str(objs[jj].item())]))
            edges[rel_id][0] = ii
            edges[rel_id][1] = jj
            if pair_key in rel_dict.keys():
                rels[rel_id] = rel_dict[pair_key]
                rel_weight[rel_id] = 4
                rel_mask_none[rel_id]=1
            else:
                rels[rel_id] = self.rel_list.index('none')
                rel_weight[rel_id] = 1
            rel_mask[rel_id] = 1
            rel_id+=1

    rel_num = rel_id
    obj_num = bound
    
    fc_feat = self.f_fc[img_id+'.jpg'][:]
    img_feat = torch.FloatTensor(np.array(fc_feat))
    return img_feat, objs, rels,rel_mask,rel_mask_none,rel_weight, edges, rel_num, obj_num, img_id , st_id

  def __len__(self):
    return len(self.ids)