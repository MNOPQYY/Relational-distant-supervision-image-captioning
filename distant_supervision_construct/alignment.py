import json
from tqdm import tqdm
from itertools import *

obj_coco = json.load(open('data/overlap_statis.json','r'))

import json
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

with open('data/dataset_coco.json','r') as input:
    coco_file = json.load(input)
name_dict={}
for img in coco_file['images']:
    filename = img['filename'].split('.')[0]
    name_dict[filename]={}
    name_dict[filename]['imgid']=img['imgid']
    split = img['split']
    if split=='restval':
        split = 'train'
    name_dict[filename]['split'] = split

with open('data/gcc_relationships.json','r') as input:
        ori_file = json.load(input)
        
with open('data/gcc_entities.json','r') as input:
    ent_corpus_data = json.load(input)
    

image_data=json.load(open('data/coco_obj_fasterrcnn_vg.json','r'))


ent_dict = ent_corpus_data['images']
    
sent_level_save={}
ent_level_save={}
noun_dict={}
final_save={}

lemmatizer  = WordNetLemmatizer()
for sent_id,sent in tqdm(enumerate(ori_file['images']),total=len(ori_file['images'])):
        save_tmp = []

        raw = sent['raw']
        relation = sent['relation']
        rel_obj=[]
        for rel in relation:
            if rel[0] not in rel_obj:
                rel_obj.append(rel[0])
            if rel[2] not in rel_obj:
                rel_obj.append(rel[2])
        noun_list = lemmatize_process(raw)
        noun_list = sorted(list(set(noun_list+rel_obj)))
        sent['full_obj'] = noun_list.copy()
        filter_noun = []
        for n in noun_list:
            if n in obj_coco:
                filter_noun.append(n)
        noun_list = '_'.join(sorted(filter_noun))
        
        if noun_list not in noun_dict.keys():
            noun_sg = []
            noun_sg.append(sent)
            noun_dict[noun_list] = noun_sg
                        
        else:
            noun_dict[noun_list].append(sent)
    
final_save['images'] = noun_dict
with open('data/setence_level_knowledge_gcc.json' ,'w') as output:
        json.dump(final_save, output)

man_list=['girl', 'woman','man','boy']
st_id = 0
img_count=[]
count=0
count_val=0
noun_find=0
count_val-=1
for ed,i in tqdm(enumerate(list(image_data.keys())),total = len((list(image_data.keys())))):
    if i not in img_count:
     if i in name_dict.keys():
      if name_dict[i]['split']=='train':
        obj_label = image_data[i]
        obj_label = list(set(obj_label).intersection(set(obj_coco)))
        if 'person' in obj_label and len(list(set(man_list).difference(set(obj_label))))<4:
            obj_label = list(set(obj_label).difference(set(['person'])))
        noun_list = sorted(list(set(obj_label)))
        
        if len(noun_list)>1:
          noun_key = '_'.join(noun_list)
          if noun_key in noun_dict.keys():
           noun_find+=1
           for sent in noun_dict[noun_key]:
            noun_list_tmp = noun_list.copy()
            if len(sent['relation'])<1:
                continue
            for rel in sent['relation']:
               s=rel[0]
               o=rel[2]
               noun_list_tmp.append(s)
               noun_list_tmp.append(o)
            noun_list_tmp = sorted(list(set(noun_list_tmp)))
            noun_list_full = sent['full_obj']
            if len(noun_list_full)-len(noun_list)>2:
                continue
            img_tmp = {}
            img_tmp['relations']=[sent.copy()]
            img_tmp['relations'][0]['align']=len(noun_list)
            img_tmp['objs']=noun_list_tmp
            
            img_tmp['sentid']=st_id
            img_tmp['filename']=i
            if i not in img_count:
                img_count.append(i)
                count+=1
                if count_val<500:
                  count_val+=1
            if count_val<500:
                split = 'val'
            else:
                split = 'train'
            img_tmp['split']=split
            sent_level_save[st_id]=img_tmp
            st_id+=1
        ent_level_save[i] = {'relations':{}}
        obj_coms = list(combinations(noun_list,2))
        for obj_com in obj_coms:
            com_key = '_'.join(sorted(obj_com))
            if com_key in ent_dict.keys():
                ent_level_save[i]['relations'][com_key] = ent_dict[com_key]
                
with open('data/entity_level_knowledge_gcc.json','w') as output:
    json.dump(ent_level_save,output)
with open('data/alignment_result_coco_gcc.json','w') as output:
    json.dump(sent_level_save,output)


