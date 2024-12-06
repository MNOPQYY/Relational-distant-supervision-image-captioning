import json
from tqdm import tqdm


with open('../data/alignment_result_coco_gcc','r') as input:
    aldata = json.load(input)

num={}
for sent in tqdm(aldata,total=len(aldata)):
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
        annotaion['train'][filename].append({'caption':raw,'from':sent['align']})
    else:
        annotaion['train'][filename]=[{'caption':raw,'from':sent['align']}]
        
json.dump(annotaion, open('./data/captions/captions.json', 'w'))

with open('../data/dataset_coco.json','r') as input:
    sent_file=json.load(input)
    
annotaion = {'test': {}}
# num={}
for sent in tqdm(sent_file['images'],total=len(sent_file['images'])):
  if sent['split'] == 'train' or sent['split'] == 'restval':
    filename=sent['filename']
    annotaion['test'][filename]=[{'caption':['empty'],'from':'align'}]
json.dump(annotaion, open('./data/captions/inf_captions.json', 'w'))