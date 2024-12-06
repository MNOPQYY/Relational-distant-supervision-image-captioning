import numpy as np
from collections import OrderedDict
import torch
import sys
# sys.path.append("coco-caption")
# sys.path.insert(0, '/home/mcislab/houjingyi/cvpr19/convcap-master/third_party/coco-caption')
sys.path.insert(0, '/home/mcislab3d/qiyayun/coco-caption-master') 
# from pyciderevalcap.ciderD.ciderD import CiderD
from pycocoevalcap.cider.cider import Cider

CiderD_scorer = None
# CiderD_scorer = CiderD(df='corpus')


def init_cider_scorer(cached_tokens):
    global CiderD_scorer
    CiderD_scorer = CiderD_scorer or Cider(df=cached_tokens)


def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()


def get_self_critical_reward(model, fc_feats, tri_feat,n_rel, gts_arr, gen_result_):
    
    batch_size = gen_result_.size(0)
    # print gen_result_[0]

    # get greedy decoding baseline
    model.eval()
    with torch.no_grad():
      _, greedy_res, _ = model(fc_feats, tri_feat, n_rel, mode='inference')
    model.train()
    # _, greedy_res = model(fc_feats, wordc_v, mode='inference')

    res = OrderedDict()
    
    # print(gen_result.shape)
    # print(greedy_res.shape)
    # _, greedy_res = model(fc_feats, mode='inference')
    

    gen_result = gen_result_.cpu().data.numpy()
    
    greedy_res = greedy_res.cpu().data.numpy()
    bs, ncap_per_img, _ = gts_arr.size()        
    gts_arr = gts_arr.cpu().data.numpy() # batchsize, ncap_per_img, 
    gts_arr = gts_arr[:,:,1:]
    
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[batch_size + i] = [array_to_str(greedy_res[i])]

    gts = OrderedDict()
    # for i in range(gts_arr.size(0)):
    # for i in range(batch_size):
        # gts[i] = [array_to_str(gts_arr[i])]
    # print gts_arr.shape
    # print gts_arr[0][0]
    for i in range(bs):
      for j in range(ncap_per_img):
        # gts[i] = [array_to_str(gts_arr[i])]
        gts[i*ncap_per_img+j] = [array_to_str(gts_arr[i][k])
                  for k in range(ncap_per_img)]
        # gts[i] = [array_to_str(gts_arr[i][j])
                  # for j in range(gts_arr.size(1))]
    # print len(gts)

    res_ = [{'image_id': i, 'caption': res[i]} for i in range(2 * batch_size)]
    gts_ = {i: gts[i % batch_size// ncap_per_img] for i in range(2 * batch_size)} 
    # gts_ = {i: gts[i % batch_size] for i in range(2 * batch_size)}
    # print gts_
    
    _, scores = CiderD_scorer.compute_score(gts_, res_)
    # print('Cider scores:', np.mean(scores[:, 0]))

    scores_ = scores[:batch_size] - scores[batch_size:]

    rewards = np.repeat(scores_[:, np.newaxis], gen_result.shape[1], 1)
    # print rewards

    return rewards
    # return gen_result

