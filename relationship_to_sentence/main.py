import argparse
import torch
import os
from train import train

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=int, default=0,help='gpu device id')
    parser.add_argument('--seed', type=int, default=586,help='random seed')
    #parameter
    parser.add_argument('--batchsize', help='batch size', default=64)
    parser.add_argument('--ncap_per_img', help='ncap', default=1)
    parser.add_argument('--max_len', help='max caption length', default=21)
    parser.add_argument('--max_obj', help='max object detection results', default=7)
    parser.add_argument('--max_rel_num', help='max relation detection results', default=7)
    parser.add_argument('--max_rel_word_num', help='max relation words number', default=4)
    parser.add_argument('--hidden_dim', help='hidden feature dim', default=1024)
    parser.add_argument('--embed_dim', help='word vector size_embedding dim', default=300)
    parser.add_argument('--decoder_dim', help='decoder dim', default=1024)
    parser.add_argument('--emb_dim', help='word vector size_embedding dim', default=300)
    
    parser.add_argument('--learning_rate', help='training learning rate', default=1e-4)#1e-4
    parser.add_argument('--lr_step_size', type=int, default=10,help='epochs to decay learning rate after')

    parser.add_argument('--dropout', help='dropout num', default=0.8)
    #traning parameter
    parser.add_argument('--epochs', help='max epoch', default=100)
    parser.add_argument('--save_epoch', help='save epoch', default=2)
    parser.add_argument('--val_after', help='which epoch to start valid', default=1)
    parser.add_argument('--score_select', help='save epoch', default='CIDEr')
    parser.add_argument('--val_test_batchsize', help='batch size', default=57)
    
    parser.add_argument('--beam_size', type=int, default=1)
    
    
    #data path    
    parser.add_argument('--model_dir', help='model save', default='./models')
    parser.add_argument('--wordlist_path', help='wordlist data path', default='data/gcc/wordlist_gcc.json')
    parser.add_argument('--data_path', help='preprocess img feat for different image with different layer', default='/home/qyy/work_space/unsupervised_img_captioning')
    
    parser.add_argument('--glove_emb_path', help='candidate triple given two objects', default='data/')
    
    return parser
    #save path

if __name__ == "__main__":
    
    parser = get_arguments()
    

    
    opt = parser.parse_args()
    
    settings = dict()
    settings['att_feat_dim'] = 512*3
    settings['d_model'] = 512  # model dim
    settings['d_ff'] = 512  # feed forward dim
    settings['h'] = 8  # multi heads num
    settings['N_enc'] = 6  # encoder layers num
    settings['N_dec'] = 6  # decoder layers num
    settings['dropout_p'] = 0.1
    settings['max_seq_len'] = 21

    opt.settings = settings
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
    print('CUDA: '+str(opt.gpu))
    torch.backends.cudnn.enabled = False
    train(opt)