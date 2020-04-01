import time
import os
from collections import defaultdict
from multiprocessing.dummy import Pool
from traceback import print_exc
import argparse

import torch
import torch.nn as nn
import numpy as np
import ujson as json
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

from Nodes import *

def create_ques_info_dict():
    # 每个question item由5个元素组成
    ques_info_dict={}
    ques_info_dict['id'] = None
    ques_info_dict['node_list'] = None
    ques_info_dict['sp_adj'] = None
    return ques_info_dict

def process(item):
    '''sub-function for multi-processes function.'''
    global tokenizer
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
    sp_adj = Adjacency_sp([])
    index_cursor = 0
    
    Q_node_cursor = index_cursor    
    Q_node = QuestionNode.build(Q_node_cursor, -1, question,
                               answer = answer, ques_type = Q_type)
    node_list.append(Q_node)
    
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
                sp_adj.append(edge_type_map['S_S'], S_node_cursor - 1 - _Entity_len, S_node_cursor)
            sp_adj.append(edge_type_map['P_S'], P_node_cursor, S_node_cursor)

            # 添加S和E之间边
            _Entity_len = len(S_node.content_NER_list)
            for entities_dict in S_node.content_NER_list:
                index_cursor += 1
                E_node_cursor = index_cursor
                E_node = EntityNode.build(E_node_cursor, S_node_cursor,
                                          entities_dict['content'])
                node_list.append(E_node)
                sp_adj.append(edge_type_map['S_E'], S_node_cursor, E_node_cursor)
              
    # in item loop
    # 连接Q节点和E节点.
    E_nodes_in_Q = [i for i in node_list if i.node_type == 'Entity' \
                    and i.content_raw.replace(' ','') in question.replace(' ','')]
    for i in E_nodes_in_Q:
        sp_adj.append(edge_type_map['Q_E'], Q_node_cursor, i.node_id)
    
    # 连接S和P节点
    S_nodes = [i for i in node_list if i.node_type == 'Sentence']
    E_nodes = [i for i in node_list if i.node_type == 'Entity']
    for E_n in E_nodes:
        entity = E_n.content_raw.replace(' ','')
        for S_n in S_nodes:
            if entity in S_n.content_raw.replace(' ',''):
                sp_adj.append(edge_type_map['S_P_hyper'], node_list[E_n.parent_id].node_id, node_list[S_n.parent_id].node_id)
            
    if len(node_list) != sp_adj.to_dense_symmetric().shape[0]:
        print(node_list)
        for i,(v_i_j) in enumerate(sp_adj.v_i_j): print(f"{i}:\t{v_i_j}")
        raise AssertionError
        
    ques_info_dict['node_list'] = node_list
    ques_info_dict['sp_adj'] = sp_adj
    return ques_info_dict

def get_cls_feature_from_LMmodel(text,text_pair=None,
                            tokenizer = None,
                            model = None,
                            add_special_tokens = True,
                           device = 'cuda',
                           test_mode = False):

    if test_mode: return torch.randn([30, 1, 768])   
    assert model
    model_input = tokenizer.encode_plus(text,text_pair,
                                        add_special_tokens=add_special_tokens,
                                        return_tensors='pt')
    
    model_input = {k:v.to(device) for k,v in model_input.items()}
    
    with torch.no_grad():
        last_hidden_state = model(**model_input)[0].cpu()[:,-1,:]
    
    del model_input
    
    return last_hidden_state

def build_save_nodes_feat(index_item, test_mode = False):
    """train on multi-gpu in multi-threadings."""
    global model_list
    index, ques_item = index_item[0], process(index_item[1])
    if os.path.exists(f"{args.save_dir}/{ques_item['id']}.json"): return
    model_index = index % len(model_list)
    model_XLNET = model_list[model_index]
        
    try:
        node_list = ques_item['node_list']
        # Q node
        Q_node = node_list[0]
        Q_node.cls_feature = get_cls_feature_from_LMmodel(Q_node.content_raw,
                                                        tokenizer = tokenizer,
                                                        model = model_XLNET,
                                                        device = DEVICE,
                                                        test_mode=test_mode) # [1,N,D]

        # S node
        for S_node in [i for i in node_list if i.node_type == 'Sentence']:

            S_node.cls_feature = get_cls_feature_from_LMmodel(Q_node.content_raw, 
                                                            S_node.content_raw,
                                                            add_special_tokens=True,
                                                            tokenizer = tokenizer,
                                                            model = model_XLNET,
                                                            device = DEVICE,
                                                            test_mode=test_mode)   

        # P node
        for P_i, P_node in [(i,n) for i,n in enumerate(node_list) if n.node_type == 'Paragraph']:
                S_in_P = [n for n in node_list if n.parent_id == P_i]
                all_S_raw = ' '.join([n.content_raw for n in S_in_P])
                P_node.cls_feature = get_cls_feature_from_LMmodel(Q_node.content_raw, 
                                                                all_S_raw,
                                                                add_special_tokens=True,
                                                                tokenizer = tokenizer,
                                                                model = model_XLNET,
                                                                device = DEVICE,
                                                                test_mode=test_mode)

        # E node
        for E_node in [i for i in node_list if i.node_type == 'Entity']:

            E_node.cls_feature = get_cls_feature_from_LMmodel(Q_node.content_raw,
                                                            E_node.content_raw, 
                                                            add_special_tokens=True,
                                                            tokenizer = tokenizer,
                                                            model = model_XLNET,
                                                            device = DEVICE,
                                                            test_mode=test_mode)

        ques_item['node_list'] = [node.to_serializable() for node in ques_item['node_list']]
        ques_item['sp_adj'] = ques_item['sp_adj'].to_serializable()

        if not os.path.exists(args.save_dir): os.mkdir(args.save_dir)
        with open(f"{args.save_dir}/{ques_item['id']}.json", 'w', encoding='utf-8') as fp:
            json.dump(ques_item, fp)
    except:
        print(index_item)
        print_exc()

def save_in_steps_multi(json_train, start = 0, end = 1000, thread_num = 1):
    thread_num = 1 if thread_num<0 else thread_num

    pbar = tqdm(total = len(json_train[start:end]), desc = f'building ndoes feature')
    index_item_list = zip(range(end - start), json_train[start:end])
    with Pool(thread_num) as pool:
        pool_iter = pool.imap(build_save_nodes_feat, index_item_list)
        for r in pool_iter:
            pbar.update()

def make_args():
    parser = argparse.ArgumentParser()

    # path information
    parser.add_argument(
        "--json_train_path",
        default='data/hotpot_train_v1.1.json',
        type=str,
        help="remain",
            )
    parser.add_argument(
        "--model_path",
        default='/g/data/models/xlnet-large-cased',
        type=str,
        help="remain",
            )
    parser.add_argument(
        "--save_dir",
        default='save_preprocess_new',
        type=str,
        help="remain",
            )
    
    # train setting.
    parser.add_argument(
        "--cuda_num",
        default=1,
        type=int,
        help="remain",
            )
    parser.add_argument(
        "--end",
        default=100,
        type=int,
        help="remain",
            )
    parser.add_argument(
        "--thread_num",
        default=1,
        type=int,
        help="remain",
            )
    parser.add_argument(
        "--spacy_model",
        default="en_core_web_sm",
        type=str,
        help="remain",
            )

    parser.add_argument("--use_mini", action="store_true", help="remain")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = make_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    set_tokenizer(tokenizer)
    set_spacy()

    with open(args.json_train_path, 'r', encoding='utf-8') as fp:
        json_train = json.load(fp)
    DEVICE_LIST = [f"cuda:{i}" for i in range(args.cuda_num)]
    model_list = []
    for DEVICE in DEVICE_LIST:
        model = AutoModel.from_pretrained(args.model_path, local_files_only=True)
        model_list.append(model.to(DEVICE))

    save_in_steps_multi(json_train=json_train, end = args.end, thread_num = args.thread_num)

"""
test:

python gen_nodes_repr.py --cuda_num 1 --end 3 --thread_num 1 \
--model_path data/models/distilbert-base-uncased-distilled-squad \
--save_dir save_test --spacy_model en_core_web_sm

formal:

python gen_nodes_repr.py --cuda_num 4 --end 3 --thread_num 1 \
--model_path data/models/distilbert-base-uncased-distilled-squad \
--save_dir save_test --spacy_model en_core_web_sm

"""

