# coding:utf8
import tqdm
import os
import time
import json
import sys
import pdb
import traceback
from bdb import BdbQuit
import numpy as np
import torch

from dataloader import get_dataloader
from models.decoder import Decoder
from opts import parse_opt
from self_critical.utils import get_ciderd_scorer, get_self_critical_reward, RewardCriterion

def train():
    opt = parse_opt()
    train_mode = opt.train_mode
    idx2word = json.load(open(opt.idx2word, 'r'))
    captions = json.load(open(opt.inf_captions, 'r'))

    decoder = Decoder(idx2word, opt.settings)
    decoder.to(opt.device)
    lr = opt.learning_rate
    optimizer, xe_criterion = decoder.get_optim_and_crit(lr)
    if opt.inf_checkpoint:
        print("====> loading checkpoint '{}'".format(opt.inf_checkpoint))
        chkpoint = torch.load(opt.inf_checkpoint, map_location=lambda s, l: s)
        assert opt.settings == chkpoint['settings'], \
            'opt.settings and resume model settings are different'
        assert idx2word == chkpoint['idx2word'], \
            'idx2word and resume model idx2word are different'
        decoder.load_state_dict(chkpoint['model'])
        if chkpoint['train_mode'] == train_mode:
            optimizer.load_state_dict(chkpoint['optimizer'])
            lr = optimizer.param_groups[0]['lr']
        print("====> loaded checkpoint '{}', epoch: {}, train_mode: {}"
              .format(opt.inf_checkpoint, chkpoint['epoch'], chkpoint['train_mode']))
    word2idx = {}
    for i, w in enumerate(idx2word):
        word2idx[w] = i
    
    test_captions = {}
    for fn in captions['test']:
        test_captions[fn] = [[]]
    test_data = get_dataloader(opt.fc_feats, opt.att_feats, test_captions, decoder.pad_id,
                               opt.max_seq_len, opt.inf_batch_size, opt.num_workers, shuffle=False)    
    results = []
    for fns, fc_feats, att_feats, _, _,_ in tqdm.tqdm(test_data, ncols=100):
        fc_feats = fc_feats.to(opt.device)
        att_feats = att_feats.to(opt.device)
        for i, fn in enumerate(fns):
            fc_feat = fc_feats[i]
            att_feat = att_feats[i]
            with torch.no_grad():
                rest, _ = decoder.sample(fc_feat, att_feat, beam_size=opt.beam_size, max_seq_len=opt.max_seq_len)
            results.append({'image_id': fn, 'caption': rest[0]})
    json.dump(results, open(os.path.join('./inference_result', 'inference_%s.json' % opt.inf_checkpoint.split('/')[-1].split('.')[0]), 'w'))


if __name__ == '__main__':
    try:
        train()
    except BdbQuit:
        sys.exit(1)
    except Exception:
        traceback.print_exc()
        print('')
        pdb.post_mortem()
        sys.exit(1)