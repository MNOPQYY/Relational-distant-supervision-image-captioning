import json
import os

from nltk.stem import WordNetLemmatizer
import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from tqdm import tqdm

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
    
with open('filter_inf_result.json','r') as input:
    ori = json.load(input)
with open('../data/entity_level_knowledge_gcc.json','r') as input:
    cand = json.load(input)['images']

sentid_dict = {}
for img in ori['annotations']:
        sent_tmp = {}
        sent_tmp['raw'] = img['caption']
        sent_tmp['image_id'] = img['image_id']
        sent_tmp['filename'] = img['filename']
        sent_tmp['objs_from_sent'] = img['objs_from_sent']
        sentid_dict[img['image_id']] = sent_tmp
print('finish building sentid dict file!')

ori_dir = 'out_sentence_scene_graph'
dire_list = os.listdir(ori_dir)

lemmatizer  = WordNetLemmatizer()
  
save_file_tr = {}
save_list_tr = []

save_file_te = {}
save_list_te = []

save_file_full = {}
save_list_full = []

img_relation_dict_train=[]
img_relation_dict_test=[]

split_count=0
sent_num = 0
for i in dire_list:
    file_list = os.listdir(ori_dir+'/'+i)
    for file in file_list:
        filename = ori_dir+'/'+i+'/'+file
        input = open(filename,'r')
        lines = input.readlines()
        for line in lines:
            line_dict = {}
            line = line.strip().replace('null','None')
            line = eval(line)
            relas = line['relationships']
            sentid = line['id']
            img_id = sentid_dict[sentid]['image_id']
            filename_img = sentid_dict[sentid]['filename']
            rela_save = []
            for rela in relas:
              s = len(rela['text'][0].split(' '))
              r = len(rela['text'][1].split(' '))
              o = len(rela['text'][2].split(' '))
              if max(s,o) == 1 and r<=4 and rela['text'] not in rela_save:
                search_k = '_'.join(sorted([rela['text'][0],rela['text'][2]]))
                if search_k in cand.keys():
                    cand_tris=cand[search_k]
                    cand_rels=[rr[1] for rr in cand_tris]
                    if rela['text'][1] in cand_rels:
                
                        rela_save.append(rela['text'])
            line_dict['raw'] = sentid_dict[sentid]['raw']
            rel_obj=[]
            for rel in rela_save:
              if rel[0] not in rel_obj:
                rel_obj.append(rel[0])
              if rel[2] not in rel_obj:
                rel_obj.append(rel[2])
            noun_list = sorted(list(set(rel_obj)))
            line_dict['rel_obj']=noun_list.copy()
            if split_count<500:
                split = 'test'
            else:
                split = 'train'
            line_dict['relation'] = rela_save
            line_dict['raw'] = sentid_dict[sentid]['raw']
            line_dict['imgid'] = sentid_dict[sentid]['image_id']
            line_dict['filename'] = sentid_dict[sentid]['filename']
            line_dict['objs_from_sent'] = sentid_dict[sentid]['objs_from_sent']
            line_dict['sentid'] = sentid
            if rela_save != []:
              save_list_full.append(line_dict)
              
              if split== 'train':
                split_count+=1
                img_relation_dict_train.append(line_dict)
              if split== 'test':
                split_count+=1
                img_relation_dict_test.append(line_dict)
        input.close()
        
save_file_tr['images'] = img_relation_dict_train
save_file_te['images'] = img_relation_dict_test
save_file_full['images'] = save_list_full

with open('coco_gcc_rela_anno_full.json','w') as output:
    json.dump(save_file_full,output)
with open('coco_gcc_rela_anno_train.json','w') as output:
    json.dump(save_file_tr,output)
with open('coco_gcc_rela_anno_val.json','w') as output:
    json.dump(save_file_te,output)