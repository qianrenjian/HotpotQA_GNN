
json_train_path = r'./data/hotpot_train_v1.1.json'
HotpotQA_path = './'
save_cache_path = 'save_cache/'
use_proxy = 1
proxies={"http_proxy": "127.0.0.1:10809",
        "https_proxy": "127.0.0.1:10809"} if use_proxy else None

import json
import torch
import torch.nn as nn
import numpy as np
import time


from transformers import AutoConfig, AutoModel, AutoTokenizer

from class_new import QuestionNode, ParagraphTitleNode, SentenceNode, EntityNode, Adjacency_sp
from collections import defaultdict
from traceback import print_exc
from tqdm import tqdm
from multiprocessing import Pool

import spacy

nlp = spacy.load("en_core_web_lg")

def find_NER_in_spacy(raw_content, nlp=nlp, tokensize=False, ner=False, \
                      exclude_list = ['PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']):
    '''使用spacy进行NER.
    标注解释: https://spacy.io/api/annotation
    '''
    res_nlp = nlp(raw_content)
    tokens = None
    if tokensize:
        tokens =  [str(i) for i in res_nlp.doc]
    
    entities_list = None
    if ner:
        entities_list = []
        for item in res_nlp.ents:
            if item.label_ in exclude_list: continue
            entities_dict = {}
            entities_dict['type'] = item.label_
            entities_dict['span_start'] = item.start
            entities_dict['content'] = item.text
            entities_dict['span_end'] = item.end
            entities_list.append(entities_dict)
        # print(dir(item))
    return tokens, entities_list

# 返回Q-paragraph(for BERT); adj; node_list
edge_type_map = {
    'Q_P':101,
    'Q_E':102,
    'P_S':103,
    'S_S_hyper':104,
    'S_E':105,
    'P_P':106,
    'S_S':107,
}

def create_ques_info_dict():
    # 每个question item由5个元素组成
    ques_info_dict={}
    ques_info_dict['id'] = None
    ques_info_dict['node_list'] = None
    ques_info_dict['sp_adj'] = None
    return ques_info_dict


def _process(item):
    '''sub-function for multi-processes function.'''
    ques_info_dict = create_ques_info_dict()

    supporting_facts = defaultdict(list)
    for s_fact in item['supporting_facts']:
        supporting_facts[s_fact[0]].append(s_fact[1])

    level = item['level']
    question = item['question']
    context = item['context']
    answer = item['answer']
    Q_id = item['_id']
    Q_type = item['type']

    ques_info_dict['id'] = Q_id

    node_list = []
    sp_adj = Adjacency_sp()
    index_cursor = 0
    
    Q_node_cursor = index_cursor    
    Q_node = QuestionNode.build(Q_node_cursor, -1, question,\
                               answer = answer, ques_type = Q_type)
    
    node_list.append(Q_node)

    ques_para_list = []
    
    for p_index, paragraph in enumerate(context):
        paragraph_label = 0

        title = paragraph[0]
        
        index_cursor += 1

        # 添加P-1和P
        try:
            if index_cursor != 0: sp_adj.append(edge_type_map['P_P'], P_node_cursor, index_cursor) 
        except (NameError): # 首次调用P_node_cursor会报错
            pass
        P_node_cursor = index_cursor
        P_node = ParagraphTitleNode.build(P_node_cursor, Q_node_cursor, title)
        
        # 判断 support paragraph
        if title in supporting_facts.keys():
            paragraph_label = 1
            P_node.set_support()
        node_list.append(P_node)
        
        # 添加Q和P
        sp_adj.append(edge_type_map['Q_P'], Q_node_cursor, P_node_cursor)

        
        for s_index, sentence in enumerate(paragraph[1]):
            S_id = f'{Q_id}_{p_index}_{s_index}'

            index_cursor += 1
            S_node_cursor = index_cursor
            S_node = SentenceNode.build(S_node_cursor, P_node_cursor, sentence)
            
            # 判断support fact
            if (paragraph_label == 1) and (s_index in supporting_facts[title]):
                S_node.set_support()
            node_list.append(S_node)

            # 添加S之间边; P和S之间边
            if s_index != 0:
                sp_adj.append(edge_type_map['S_S'], S_node_cursor - 1, S_node_cursor)
            sp_adj.append(edge_type_map['P_S'], P_node_cursor, S_node_cursor)

            # 添加S和E之间边
            for entities_dict in S_node.content_NER_list:
                index_cursor += 1
                E_node_cursor = index_cursor
                E_node = EntityNode.build(E_node_cursor, S_node_cursor, \
                                          entities_dict['content'])
                E_node.set_span_in_sentence(entities_dict['span_start'])

                node_list.append(E_node)
                sp_adj.append(edge_type_map['S_E'], S_node_cursor, E_node_cursor)
              
    # in item loop
    # 连接Q节点和E节点.
    E_nodes_in_Q = [i for i in node_list if i.node_type == 'Entity' and i.content_raw.replace(' ','') \
                    in question.replace(' ','')]
    for i in E_nodes_in_Q:
        sp_adj.append(edge_type_map['Q_E'], Q_node_cursor, i.node_id)
    
    # 连接S和P节点
    S_node = [i for i in node_list if i.node_type == 'Sentence']
    E_nodes = [i for i in node_list if i.node_type == 'Entity']
    for E_n in E_nodes:
        entity = E_n.content_raw.replace(' ','')
        for S_n in S_node:
            if entity in S_n.content_raw.replace(' ',''):
                sp_adj.append(edge_type_map['S_S_hyper'], node_list[E_n.parent_id].node_id, node_list[S_n.parent_id].node_id)
            
    ques_info_dict['node_list'] = node_list
    ques_info_dict['sp_adj'] = sp_adj
    return ques_info_dict




def preprocessing(item_num = 2, process_num = 1):
    '''main multi-processes function.'''
    item_num = None if item_num<0 else item_num
    process_num = 1 if process_num<0 else process_num
    
    resturn_list = []
    pbar = tqdm(total = len(json_train[:item_num]), desc = f'processing json items')
    with Pool(process_num) as pool:
        pool_iter = pool.imap(_process, json_train[:item_num])
        for i,r in enumerate(pool_iter):
            resturn_list.append(r)
            pbar.update()
    return resturn_list


hotpotQA_train_preprocess = preprocessing(128,16)

Hotpot_index_items = [(i+1, item) for i,item in enumerate(hotpotQA_train_preprocess)]

def get_features_from_XLNET(text,text_pair=None,
                            tokenizer = None,
                            model = None,
                            add_special_tokens = True,
                           device = 'cuda'):
    '''XLNET在512张TPU v3上训练5.5天得到. 一张TPU 8核心 128GB内存.'''
    
    assert model
    model_input = tokenizer_XLNET.encode_plus(text,text_pair,
                                        add_special_tokens=add_special_tokens,
                                        return_tensors='pt')
    
    model_input = {k:v.to(device) for k,v in model_input.items()}
    
    # 不能在函数里面设置device.
    # model.to(device)
    with torch.no_grad():
        last_hidden_state = model(**model_input)[0].cpu()
    
    del model_input
    
    return last_hidden_state



model_name = 'xlnet-base-cased'

proxies={"http_proxy": "127.0.0.1:10809",
         "https_proxy": "127.0.0.1:10809"}
proxies=None

tokenizer_XLNET = AutoTokenizer.from_pretrained(model_name,proxies=proxies)

model_path = '/g/data/models/xlnet-base-cased'

model_XLNET01 = AutoModel.from_pretrained(model_path,local_files_only=True)
model_XLNET01.eval()
_ = model_XLNET01.to(DEVICE)


def _build(index_item):
    index, ques_item = index_item[0], index_item[1]
    model_XLNET = model_XLNET01

    try:
        node_list = ques_item['node_list']
        # Q node
        Q_node = node_list[0]
        Q_node.cls_feature = get_features_from_XLNET(Q_node.content_raw,
                                                         tokenizer = tokenizer_XLNET,
                                                          model = model_XLNET,
                                                         device = DEVICE)[:,-1,:] # [1,N,D]

        # S node
        for S_node in [i for i in node_list if i.node_type == 'Sentence']:
            S_node.cls_feature = get_features_from_XLNET(Q_node.content_raw, 
                                                            S_node.content_raw,
                                                            add_special_tokens=True,
                                                            tokenizer = tokenizer_XLNET,
                                                            model = model_XLNET,
                                                            device = DEVICE)[:,-1,:]    

        # P node
        for P_i, P_node in [(i,n) for i,n in enumerate(node_list) if n.node_type == 'Paragraph']:
                S_in_P = [n for n in node_list if n.parent_id == P_i]
                all_S_raw = ' '.join([n.content_raw for n in S_in_P])
                P_node.cls_feature = get_features_from_XLNET(Q_node.content_raw, 
                                                                all_S_raw,
                                                                add_special_tokens=True,
                                                                tokenizer = tokenizer_XLNET,
                                                                model = model_XLNET,
                                                                device = DEVICE)[:,-1,:]

        # E node
        for E_node in [i for i in node_list if i.node_type == 'Entity']:

            E_node.cls_feature = get_features_from_XLNET(Q_node.content_raw,
                                                            E_node.content_raw, 
                                                            add_special_tokens=True,
                                                            tokenizer = tokenizer_XLNET,
                                                            model = model_XLNET,
                                                            device = DEVICE)[:,-1,:]


        return ques_item
    except:
        print_exc()



# torch.multiprocessing.set_start_method('spawn', force=True)
# model_XLNET.share_memory()

def multi_build(from_index = 0, to_index = None, thread_num = 1):
    try:
        thread_num = 1 if thread_num<0 else thread_num

        resturn_list = []
        pbar = tqdm(total = len(Hotpot_index_items[from_index:to_index]), desc = f'building features')
        with Pool(thread_num) as pool:
            pool_iter = pool.imap(_build, Hotpot_index_items[from_index:to_index])
            for i,r in enumerate(pool_iter):
                resturn_list.append(r)
                pbar.update()
        return resturn_list
    except:
        print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print_exc()

hotpotQA_preprocess_cls = multi_build(-1,1)





if __name__ == '__main__':
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open(json_train_path, 'r', encoding='utf-8') as fp:
        json_train = json.load(fp)
        


    tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
    def tokensizer_in_Model(content_raw, special=False, tokenizer = tokenizer):
        tokens = tokenizer.tokenize(content_raw)
        return tokens



    








































