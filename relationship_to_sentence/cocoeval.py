"""
From Karpathy's neuraltalk2:
https://github.com/karpathy/neuraltalk2
Specifically:
https://github.com/karpathy/neuraltalk2/blob/master/coco-caption/myeval.py
"""

import sys

import numpy as np
import os
import os.path as osp
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')
import sys


def language_eval(input_data, savedir, split, is_beam = False):
  use_scorer=['Bleu', 'METEOR', 'ROUGE_L', 'CIDEr']
  if type(input_data) == str: # Filename given.
    checkpoint = json.load(open(input_data, 'r'))
    preds = checkpoint
  elif type(input_data) == list: # Direct predictions give.
    preds = input_data

  annFile = 'data/gcc_'+split+'_eval_file.json'
  coco = COCO(annFile)
  valids = coco.getImgIds()

  # Filter results to only those in MSCOCO validation set (will be about a third)
  preds_filt = [p for p in preds if p['image_id'] in valids]
  print ('Using %d/%d predictions' % (len(preds_filt), len(preds)))
  resFile = osp.join(savedir, 'result_%s.json' % (split))
  preds_filt_str = []
  
  for ii in range(len(preds_filt)):
    str_imgid = preds_filt[ii]['image_id']
    str_imgid = int(str(str_imgid.numpy()))
    preds_filt_str.append({'image_id': str_imgid, 'caption': preds_filt[ii]['caption']})
  json.dump(preds_filt_str, open(resFile, 'w')) # Serialize to temporary json file. Sigh, COCO API...
  
  cocoRes = coco.loadRes(resFile)
  cocoEval = COCOEvalCap(coco, cocoRes, tokenizer=None, use_scorers=use_scorer)
  cocoEval.params['image_id'] = cocoRes.getImgIds()
  cocoEval.evaluate()

  # Create output dictionary.
  out = {}
  for metric, score in cocoEval.eval.items():
    out[metric] = score
    
  if out['CIDEr'] > 3.9:
    resFileB = osp.join(savedir, 'result_%s_%f.json' % (split, out['CIDEr']))
    os.system('cp %s %s' % (resFile, resFileB))


  # Return aggregate and per image score.
  return out, cocoEval.evalImgs
