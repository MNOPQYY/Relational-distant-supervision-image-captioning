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
#MSCOCO Karpathy split: train 87783 restval 30504 val 5000 test 5000

# from PIL import Image

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms
'''
def read_binary_blob(filepath):
    file = open(filepath,'rb')
    try:
        feat = file.read()
        #num*channel*length*height*width
        (n,c,l,h,w) = array.array('i', feat[:20])
        feature_vec = np.array(array.array('f', feat[20:]))
        print (shape(feature_vec))
    finally:
        file.close()
        return feature_vec
'''

class inf_loader(Dataset):
  """Loads train/val/test splits of coco dataset"""

  def __init__(self, args, split='inference'):
    self.split = split
    self.max_rel_num = args.max_rel_num
    self.max_rel_word_num = args.max_rel_word_num
    self.max_obj_num = args.max_obj
    self.max_cand_num = args.max_cand_num
    
    self.f_fc = h5py.File(args.img_feature_path,'r')
    anno_name = args.full_coco_path
    print('Loading annotation file...')
    with open(anno_name,'r') as input:
        anno_file = json.load(input)
    annos = {}
    for img_id, img in enumerate(list(anno_file.keys())):
            annos[img] = anno_file[img]
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
    img_id = self.ids[idx]
    anno = self.annos[img_id]
    img_id = img_id.split('_b')[0]
    obj_img_result = anno
    obj_img_result = list(set(obj_img_result).intersection(set(self.objdetlist)))
    person_list=['girl', 'woman','man','person','boy']
    man_list=['girl', 'woman','man','boy']
    if 'person' in obj_img_result and len(list(set(man_list).difference(set(obj_img_result))))<4:
      obj_img_result = list(set(obj_img_result).difference(set(['person'])))
    obj_img_result = sorted(obj_img_result)
    
    rel_mask = torch.FloatTensor(self.max_rel_num).zero_()
    rels = torch.LongTensor(self.max_rel_num).zero_()
    edges = torch.LongTensor(self.max_rel_num, 2).zero_()
    objs = torch.LongTensor(self.max_obj_num).zero_()    
    cand_rels = torch.zeros(self.max_rel_num,self.max_cand_num).long()
    cand_rels[:,:] = 2
    cand_rels_num = torch.zeros(self.max_rel_num).long()
    
    obj_dict = {}
    bound=min(len(obj_img_result), self.max_obj_num)
    obj_count = 0
    obj_dict = {}
    for ii in range(bound):
            word = obj_img_result[ii]
            if(word not in self.obj_list):
                word = 'unknown'
            obj_idx = self.obj_list.index(word)
            objs[ii]=obj_idx
    rel_id = 0
    for ii in range(bound):
        if rel_id==self.max_rel_num:
            break
        for jj in range(ii,bound):
            if ii == jj:
                continue
            edges[rel_id,0]=ii
            edges[rel_id,1]=jj
            rel_mask[rel_id] = 1
            rel_id+=1
            if rel_id==self.max_rel_num:
                break
    rel_num = min(rel_id,self.max_rel_num)
    obj_num = bound
    fc_feat = self.f_fc[img_id+'.jpg'][:]
    img_feat = torch.FloatTensor(np.array(fc_feat))
    return img_feat, objs, rels,rel_mask, edges, rel_num, obj_num, img_id 
   
  def __len__(self):
    return len(self.ids)