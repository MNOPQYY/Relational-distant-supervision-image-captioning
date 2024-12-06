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

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


class coco_loader(Dataset):
  """Loads train/val/test splits of coco dataset"""

  def __init__(self, args, split='train', max_tokens=20, ncap_per_img=5):
    self.max_tokens = max_tokens
    self.ncap_per_img = ncap_per_img
    self.split = split
    #Splits from http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
    
    self.max_rel_num = args.max_rel_num
    self.max_rel_word_num = args.max_rel_word_num
    self.max_obj_num = args.max_obj
    
    anno_name = 'data/gcc_relationships_'+self.split+'_addnone.json'
    self.get_split_info(anno_name)
    
    worddict_tmp = json.load(open(args.wordlist_path, 'r'))
    worddict_tmp = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"] + sorted(worddict_tmp)
    self.wordlist = worddict_tmp
    self.numwords = len(self.wordlist)
    print('[DEBUG] #words in wordlist: %d' % (self.numwords))
    
    obj_dict_tmp = json.load(open(args.glove_emb_path+'gcc_obj_list.json', 'r'))
    self.obj_list = obj_dict_tmp
    self.numwords_obj = len(self.obj_list)
    print('[DEBUG] #words in object wordlist: %d' % (self.numwords_obj))
    
    rel_dict_tmp = json.load(open(args.glove_emb_path+'gcc_pred_list.json', 'r'))
    self.rel_list = rel_dict_tmp
    self.numwords_rel = len(self.rel_list)
    print('[DEBUG] #words in relation wordlist: %d' % (self.numwords_rel))
    
    

  def get_split_info(self, split_file):
    print('Loading annotation file...')
    with open(split_file) as fin:
      split_info = json.load(fin)
    annos = {}
    for item in split_info['images']:
          annos[item['sentid']] = item
      
    self.annos = annos
    self.ids = list(self.annos.keys())
    print('Found %d images in split: %s'%(len(self.ids), self.split))

  def __getitem__(self, idx):    
    img_id = self.ids[idx]
    anno = self.annos[img_id]
    captions = anno['raw']
    relations = anno['relation']
    rels = torch.LongTensor(self.max_rel_num).zero_()
    edges = torch.LongTensor(self.max_rel_num, 2).zero_()
    objs = torch.LongTensor(self.max_obj_num).zero_()
    rel_word_nums = torch.ones(self.max_rel_num).long()
    obj_dict = {}
    obj_num = 0
    rel_flag=0
    for rel_id,rel in enumerate(relations):
      if rel_id<self.max_rel_num:
       

        words=sorted([rel[0],rel[2]])
        obj_dict_list=[]
        for word_id,word in enumerate(words):
         if(word not in self.obj_list):
            word = 'unknown'
         obj_id = self.obj_list.index(word)
         if obj_id not in obj_dict.keys():
            if obj_num>=self.max_obj_num:
                rel_flag=1
                continue
            obj_dict[obj_id] = obj_num
            obj_num+=1
         obj_dict_list.append(obj_id)
        if rel_flag==1:
            rel_flag=0
            continue
       
        loc=1
        word = rel[loc]
        if(word not in self.rel_list):
            word = 'unknown'
        rels[rel_id]=self.rel_list.index(word)
        
        edges[rel_id,0] = obj_dict[obj_dict_list[0]]
        edges[rel_id,1] = obj_dict[obj_dict_list[1]]
        
    
    for ii in range(min(obj_num, self.max_obj_num)):
        objs[ii] = list(obj_dict.keys())[ii]
    rel_num = min(len(relations),self.max_rel_num)
    obj_num = min(obj_num,self.max_obj_num)
    

    captions = [captions]


    wordclass = torch.LongTensor(len(captions), self.max_tokens).zero_()
    sentence_mask = torch.ByteTensor(len(captions), self.max_tokens).zero_()

    for i, caption in enumerate(captions):
      table = str.maketrans(dict.fromkeys(string.punctuation))
      words = str(caption).lower().translate(table).strip().split()
      words = ['<SOS>'] + words + ['<EOS>']
      num_words = min(len(words), self.max_tokens-1)
      sentence_mask[i, :(num_words+1)] = 1
      for word_i, word in enumerate(words):
        if(word_i >= num_words):
          break
        if(word not in self.wordlist):
          word = '<UNK>'
        wordclass[i, word_i] = self.wordlist.index(word)
        
    return objs, rels, edges, rel_word_nums, rel_num, obj_num, captions, wordclass, sentence_mask, img_id 
    

  def __len__(self):
    return len(self.ids)
    # return 500