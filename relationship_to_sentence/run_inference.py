import argparse
import torch
import os
from inference import inference

from atts2vt import DecoderRNN, EncoderRNN, S2VTAttModel, EncoderPool
from graph import GraphTripleConvNet
from atts2vt.ShareEmbedding import Share_Embedding

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0,help='gpu device id')
    parser.add_argument('--seed', type=int, default=586,help='random seed')
    
    parser.add_argument('--best_modelpth', type=str, default='bestmodel.pth',help='which step to run')
    parser.add_argument('--input_name', type=str, default='result.json',help='which step to run')
    parser.add_argument('--out_name', type=str, default='output.json',help='which step to run')
    
    
    #parameter
    parser.add_argument('--batchsize', type=int, help='batch size', default=1)
    parser.add_argument('--ncap_per_img', help='ncap', default=1)
    parser.add_argument('--max_obj', help='max object detection results', default=10)
    parser.add_argument('--max_rel_num', help='max relation detection results', default=50)
    parser.add_argument('--max_rel_word_num', help='max relation words number', default=4)
    parser.add_argument('--hidden_dim', help='hidden feature dim', default=1024)
    parser.add_argument('--decoder_dim', help='decoder dim', default=1024)
    parser.add_argument('--emb_dim', help='word vector size_embedding dim', default=300)
    
    parser.add_argument('--learning_rate', help='training learning rate', default=1e-4)#1e-4
    parser.add_argument('--lr_step_size', type=int, default=10,help='epochs to decay learning rate after')
    
    parser.add_argument('--model_dir', help='model save', default='./models')
    parser.add_argument('--wordlist_path', help='wordlist data path', default='data/wordlist_gcc.json')
    parser.add_argument('--glove_emb_path', help='candidate triple given two objects', default='data/')
    
    return parser

if __name__ == "__main__":
    
    parser = get_arguments()
    

    
    args = parser.parse_args()
    
    settings = dict()
    settings['att_feat_dim'] = 512*3
    settings['d_model'] = 512  # model dim
    settings['d_ff'] = 512  # feed forward dim
    settings['h'] = 8  # multi heads num
    settings['N_enc'] = 6  # encoder layers num
    settings['N_dec'] = 6  # decoder layers num
    settings['dropout_p'] = 0.1
    settings['max_seq_len'] = 21

    args.settings = settings
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print('CUDA: '+str(args.gpu))
    torch.backends.cudnn.enabled = False
    
    inference(args, 'inference')