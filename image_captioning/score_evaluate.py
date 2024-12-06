"""
From Karpathy's neuraltalk2:
https://github.com/karpathy/neuraltalk2
Specifically:
https://github.com/karpathy/neuraltalk2/blob/master/coco-caption/myeval.py
"""

import numpy as np
import os
import os.path as osp
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')
import sys


def language_eval(num,input_data, savedir, split, annFile, is_beam = False):
  use_scorer=['Bleu', 'METEOR', 'ROUGE_L', 'CIDEr']
  if type(input_data) == str: 
    checkpoint = json.load(open(input_data, 'r'))
    preds = checkpoint
  elif type(input_data) == list: 
    preds = input_data

  annFile = './data/eval_data/coco_test.json'
  coco = COCO(annFile)
  valids = coco.getImgIds()
  preds_filt = [p for p in preds if p['image_id'] in valids]
  print ('Using %d/%d predictions' % (len(preds_filt), len(preds)))
  resFile = osp.join(savedir, 'result_%s.json' % (str(num)))
  preds_filt_str = []
  
  for ii in range(len(preds_filt)):
    str_imgid = preds_filt[ii]['image_id']
    preds_filt_str.append({'image_id': str_imgid, 'caption': preds_filt[ii]['caption']})
  json.dump(preds_filt_str, open(resFile, 'w'))
  cocoRes = coco.loadRes(resFile)
  cocoEval = COCOEvalCap(coco, cocoRes, tokenizer=None, use_scorers=use_scorer)
  cocoEval.params['image_id'] = cocoRes.getImgIds()
  cocoEval.evaluate()

  out = {}
  for metric, score in cocoEval.eval.items():
    out[metric] = score
    
  if is_beam and out['CIDEr'] > 0.74:
    resFileB = osp.join(savedir, 'result_%s_%f.json' % (split, out['CIDEr']))
    os.system('cp %s %s' % (resFile, resFileB))
  for k, v in out.items():
        print('%s %f ' % (k, v))
  f = open(osp.join('./result/eval/score'+split+'.txt'),'a')
  f.write('file:'+' '+str(input_data)+'\n')
  for k, v in out.items():
        print('%s %f ' % (k, v))
        f.write(k+' '+str(v)+'\n')
  f.write('\n')
  f.close()
  return out, cocoEval.evalImgs

if __name__ == '__main__': 
    path_ori = './result/xe/val_result/'
    path_list = os.listdir(path_ori)
    for path in sorted(path_list):
      if path.split('.')[-1]=='json':
        p=path_ori+path
        num=path.split('.')[0].split('_')[-1]
        language_eval(num, p, './result/eval/val', 'val', './data/eval_data/coco_val.json', is_beam = False)
        
    path_ori = './result/xe/'
    path_list = os.listdir('./result/xe/')
    for path in sorted(path_list):
      if path.split('.')[-1]=='json':
        p=path_ori+path
        num=path.split('.')[0].split('_')[-1]
        language_eval(num, p, './result/eval', 'test', './data/eval_data/coco_test.json', is_beam = False)
    