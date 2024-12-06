import json
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
import string
import argparse

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None
        
def lemmatize_process(sent):
    word_lemma = []
    for word, pos in pos_tag(word_tokenize(sent)):
        wordnet_pos = get_wordnet_pos(pos) or None
        if wordnet_pos == wordnet.NOUN:
            res = lemmatizer.lemmatize(word, pos=wordnet_pos)
            if res not in word_lemma:
                word_lemma.append(res)
    return word_lemma
lemmatizer  = WordNetLemmatizer()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='input_file.json',help='which step to run')
    return parser

if __name__ == "__main__":
    
    parser = get_arguments()
    args = parser.parse_args()

    obj_coco = json.load(open('../data/overlap_statis.json','r'))
    coco = json.load(open('../distant_supervision_construct/data/dataset_coco.json','r'))
    pseudo_captions = json.load(open(args.path ,'r'))
    obj_det = json.load(open('../distant_supervision_construct/data/coco_obj_fasterrcnn_vg.json','r'))
    
    save_list = []
    name_dict={}
    sentid=0
    for img in coco['images']:
        filename = img['filename'].split('.')[0]
        name_dict[filename]={}
        name_dict[filename]['imgid']=img['cocoid']
        split = img['split']
        if split=='restval':
            split = 'train'
        name_dict[filename]['split'] = split
        if split!='train':
            save_list.append({'imgid':name_dict[filename]['imgid'],'split':name_dict[filename]['split'],'filename':filename,'raw':'empty','sentid':sentid,'from':'align'})
            sentid+=1
    for img in tqdm(pseudo_captions,total=len(pseudo_captions)):
        filename = img['image_id']
        if name_dict[filename]['split']=='train':
            if img['caption']!='':
                objs = list(set(obj_det[filename]))
                objs = list(set(objs).intersection(set(obj_coco)))
                raw=img['caption']
                words = nltk.word_tokenize(sent)
                tags = nltk.pos_tag(words)
                word_list=[]
                for word,pos in tags:
                    if pos in ["NN","NNS","VB","VBD","VBG","VBN","VBP","VBZ"]:
                        word_list.append(word)
                table = str.maketrans(dict.fromkeys(string.punctuation))
                words = str(raw).lower().translate(table)
                noun_list = lemmatize_process(words)
                noun_int = list(set(objs).intersection(set(noun_list)))
                if len(noun_int)>0 and len(word_list)-len(list(set(word_list)))<1:
                    tmp={'imgid':name_dict[filename]['imgid'],'split':name_dict[filename]['split'],'filename':filename,'raw':img['caption'],'sentid':sentid,'from':'gen'}
                    save_list.append(tmp)
                    sentid+=1
            
        
    num={}
    for sent in tqdm(save_list,total=len(save_list)):
      if sent['split'] == 'train':
        if sent['filename'] not in num.keys():
            num[sent['filename']]=[0,0]
        if num[sent['filename']][1]==8:
            num[sent['filename']][0]+=1
            num[sent['filename']][1]=0
        filename=sent['filename']+'.jpg*'+str(num[sent['filename']][0])
        num[sent['filename']][1]+=1
        raw=sent['raw'].split(' ')
        if filename in annotaion['train'].keys():
            annotaion['train'][filename].append({'caption':raw,'from':'gen'})
        else:
            annotaion['train'][filename]=[{'caption':raw,'from':'gen'}]
    json.dump(annotaion, open('./data/captions/captions.json', 'w'))
    
  