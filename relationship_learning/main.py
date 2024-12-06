import argparse
import torch
import os
from training import train

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=int, default=1,help='gpu device id')
    parser.add_argument('--seed', type=int, default=586,help='random seed')
    parser.add_argument('--batchsize', help='batch size', default=32)
    parser.add_argument('--max_obj', help='max object detection results', default=7)
    parser.add_argument('--max_rel_num', help='max relation detection results', default=21)
    parser.add_argument('--max_rel_word_num', help='max relation words number', default=4)
    parser.add_argument('--max_cand_num', help='max relation words number', default=20)
    parser.add_argument('--hidden_dim', help='hidden feature dim', default=1024)
    parser.add_argument('--emb_dim', help='word vector size_embedding dim', default=300)
    parser.add_argument('--decoder_dim', help='decoder dim', default=1024)
    parser.add_argument('--feat_dim', help='decoder dim', default=2048)
    parser.add_argument('--learning_rate', help='training learning rate', default=5e-5)
    parser.add_argument('--lr_step_size', type=int, default=10,help='epochs to decay learning rate after')
    parser.add_argument('--dropout', help='dropout num', default=0.8)
    #traning parameter
    parser.add_argument('--epochs', help='max epoch', default=100)
    parser.add_argument('--save_epoch', help='save epoch', default=2)
    parser.add_argument('--val_after', help='which epoch to start valid', default=1)
    
    parser.add_argument('--model_dir', help='model save', default='./models')
    parser.add_argument('--img_feature_path', help='path for image features', default='../distant_supervision_construct/BUTD_model-master/data/coco_resnet101/coco_fc.h5')
    parser.add_argument('--obj_list_path', help='path for object categories', default='../distant_supervision_construct/data/overlap_statis.json')
    parser.add_argument('--sent_align_path', help='scene level knowledge for training relationship learning module', default='../distant_supervision_construct/data/coco_gcc_rela_anno_')
    parser.add_argument('--glove_emb_path', help='candidate triple given two objects', default='./preprocess/')
    
    return parser

if __name__ == "__main__":
    
    parser = get_arguments()
    opt = parser.parse_args()
    
    settings = dict()
    settings['d_model'] = 1024  # model dim
    settings['d_ff'] = 2048  # feed forward dim
    settings['h'] = 8  # multi heads num
    settings['N_enc'] = 2  # encoder layers num
    settings['N_dec'] = 2  # decoder layers num
    settings['dropout_p'] = 0.1

    opt.settings = settings
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
    print('CUDA: '+str(opt.gpu))
    train(opt)