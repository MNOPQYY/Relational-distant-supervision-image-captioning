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
from models import Similarity_Measure, Model_M, LanguageModelCriterion
from tqdm import tqdm 
from val import valtest 
from inference import inference


def train(args):

  t_start = time.time()
  print('[DEBUG] Loading train data ... %f secs' % (time.time() - t_start))
  
  sim_model = Similarity_Measure(args)
  sim_model.cuda()
  sim_model.train(True)
  
  model_M = Model_M(args, sim_model)
  model_M.cuda()
  model_M.train(True)
  
  m_optimizer = optim.Adam(model_M.parameters(), lr=args.learning_rate, weight_decay=0)
  exp_lr_scheduler = lr_scheduler.StepLR(m_optimizer, step_size=args.lr_step_size, gamma=.8)
  batchsize = args.batchsize
  bestscore = .0
  
  bestmodelfn = osp.join(args.model_dir, args.bestmodel)
  if(osp.exists(bestmodelfn)):
    print('loading bestmodel...')
    modelfn = torch.load(bestmodelfn)
    sim_model.load_state_dict(modelfn['sim_dict'])
    model_M.load_state_dict(modelfn['m_dict'])
    
    model_M.calculate_sim = sim_model
    
  obj_g_e = np.load(args.glove_emb_path+'obj_glove_emb.npy')
  rel_g_e = np.load(args.glove_emb_path+'pred_glove_emb.npy')
  inference(args,split='inference',model_M = model_M)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0,help='gpu device id')
    parser.add_argument('--seed', type=int, default=586,help='random seed')
    parser.add_argument('--out_name', type=str, default='coco_inference_result.h5',help='output filename')
    parser.add_argument('--bestmodel', type=str, default='bestmodel.pth',help='bestmodel filename')
    parser.add_argument('--batchsize', type=int, help='batch size', default=100)
    parser.add_argument('--ncap_per_img', help='ncap', default=1)
    parser.add_argument('--max_obj', help='max object detection results', default=7)
    parser.add_argument('--max_rel_num', help='max relation detection results', default=21)
    parser.add_argument('--max_rel_word_num', help='max relation words number', default=4)
    parser.add_argument('--max_cand_num', help='max relation words number', default=20)
    parser.add_argument('--hidden_dim', help='hidden feature dim', default=1024)
    parser.add_argument('--emb_dim', help='word vector size_embedding dim', default=300)
    parser.add_argument('--decoder_dim', help='decoder dim', default=1024)
    parser.add_argument('--feat_dim', help='decoder dim', default=2048)
    parser.add_argument('--learning_rate', help='training learning rate', default=1e-4)
    parser.add_argument('--lr_step_size', type=int, default=10,help='epochs to decay learning rate after')
    parser.add_argument('--dropout', help='dropout num', default=0.8)
    parser.add_argument('--epochs', help='max epoch', default=100)
    parser.add_argument('--save_epoch', help='save epoch', default=2)
    parser.add_argument('--val_after', help='which epoch to start valid', default=1)
    parser.add_argument('--model_dir', help='model save', default='./models')
    parser.add_argument('--img_feature_path', help='path for image features', default='../distant_supervision_construct/BUTD_model-master/data/coco_resnet101/coco_fc.h5')
    parser.add_argument('--glove_emb_path', help='embeddings for objects and predicts', default='./preprocess/')
    parser.add_argument('--obj_list_path', help='path for object categories', default='../distant_supervision_construct/data/overlap_statis.json')
    parser.add_argument('--full_coco_path', help='coco object detection result', default='../distant_supervision_construct/data/coco_obj_fasterrcnn_vg.json')
    
    return parser

if __name__ == "__main__":
    
    parser = get_arguments()
    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
    print('CUDA: '+str(opt.gpu))
    train(opt)