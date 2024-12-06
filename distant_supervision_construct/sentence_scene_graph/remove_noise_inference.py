import json
from nltk.stem import WordNetLemmatizer
import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from tqdm import tqdm
import string
import random

obj_coco = json.load(open('../data/overlap_statis.json','r'))
person_list=['girl', 'woman','man','person','boy']
man_list=['girl', 'woman','man','boy']

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
    # tag = []
    for word, pos in pos_tag(word_tokenize(sent)):
        wordnet_pos = get_wordnet_pos(pos) or None
        if wordnet_pos == wordnet.NOUN:
            res = lemmatizer.lemmatize(word, pos=wordnet_pos)
            if res not in word_lemma:
                word_lemma.append(res)
    return word_lemma
    
lemmatizer  = WordNetLemmatizer()

with open('../data/dataset_coco.json','r') as input:
    sent_file=json.load(input)

filename_dict={}
for xx in tqdm(sent_file['images'],total=len(sent_file['images'])):
  imgid=xx['imgid']
  fn = xx['filename']
  filename_dict[fn]=imgid

with open('../BUTD_model-master/inference_result/inference_result.json','r') as input:
    ori=json.load(input)
    
obj_det =json.load(open('../data/coco_obj_fasterrcnn_vg.json','r'))
    
save_file=[]
for img in ori:
    key=img['image_id']
    imgid=filename_dict[key]
    img['filename'] = key.split('.')[0]
    img['image_id'] = imgid
    objs = list(set(obj_det[key.split('.')[0]]).intersection(set(obj_coco)))
    if 'person' in objs and len(list(set(man_list).difference(set(objs))))<4:
            objs = list(set(objs).difference(set(['person'])))
    if list(set(objs).difference(set(person_list)))==[]:
        continue
    raw=img['caption']
    table = str.maketrans(dict.fromkeys(string.punctuation))
    words = str(raw).lower().translate(table)
    noun_list = lemmatize_process(words)
    img['objs_from_sent']=list(set(noun_list))
    noun_int = list(set(objs).intersection(set(noun_list)))
    if (len(noun_int)>=len(objs)-1 and len(objs)>1) or (len(noun_int)>=len(objs)-2 and len(objs)>2):
        save_file.append(img) 

with open('filter_inf_result.json','w') as output:
    json.dump({'annotations':save_file},output)