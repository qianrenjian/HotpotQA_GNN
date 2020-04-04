import os
import re
from itertools import permutations
from traceback import print_exc

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

import ujson as json
from tqdm import tqdm

from Nodes import Adjacency_sp, auto_reload_Node

class HotpotQA_GNN_Dataset(Dataset):
    def __init__(self, train_list, val_list):
        
        # elements
        self.train_list = train_list # json files path list.
        self.train_size = len(self.train_list)
        
        self.val_list = val_list
        self.val_size = len(self.val_list)
        
        self.max_node_num = -1
        
        # func
        self._lookup_dict = {'train': (self.train_list, self.train_size),
                    'val': (self.val_list, self.val_size)}

        self.set_split('train')

        # parameters
        self.pad_max_num = 0
        self.pad_value = 0
        self.pad_to_max_length = True
    
    @classmethod
    def build_dataset(cls, hotpotQA_item_folder = 'save_preprocess_new',
                      i_from = 0, i_to = 1000, ratio_train=0.7, seed=123):
        hotpotQA_item_path_list = os.listdir(hotpotQA_item_folder)
        hotpotQA_item_path_list = [f'{hotpotQA_item_folder}/{item_path}' for item_path in hotpotQA_item_path_list]
        hotpotQA_item_list = [HotpotQA_GNN_Dataset._rebuild(path) \
                              for path in tqdm(hotpotQA_item_path_list[i_from:i_to], 
                                               desc = f"loading {i_from}~{i_to}")]
                            
        np.random.seed(seed)
        np.random.shuffle(hotpotQA_item_list)        
        
        sample_train_num = int(ratio_train * len(hotpotQA_item_list))

        train_list = hotpotQA_item_list[:sample_train_num]
        val_list = hotpotQA_item_list[sample_train_num:]

        return cls(train_list, val_list)

    @staticmethod
    def _rebuild(path):
        with open(path, 'r', encoding = 'utf-8') as fp:
            QA_item = json.load(fp)
            
        QA_item['node_list'] = [auto_reload_Node(state_dicts) for state_dicts in QA_item['node_list']]
        QA_item['sp_adj'] = Adjacency_sp.from_serializable(QA_item['sp_adj'])
        return QA_item
    
    @staticmethod
    def get_weights(device = 'cpu'):
        # statics of QA dataset
        total_sent = float(3703344)
        support_sent = float(215684)
                
        total_para = float(899667)
        support_para = float(180894)
        
        total_ques = 90447
        yesno_ques = 5481 # 1 是 yesno
        
        # class 1 is always the less one.
        class_weights_sent = torch.tensor([support_sent/total_sent
                                                ,1 - support_sent/total_sent], device=device)
        
        class_weights_para = torch.tensor([support_para/total_para
                                                ,1 - support_para/total_para], device=device)
        
        class_weights_Qtype = torch.tensor([yesno_ques/total_ques
                                                ,1 - yesno_ques/total_ques], device=device)
        
        return class_weights_sent, class_weights_para, class_weights_Qtype
    
    def set_parameters(self, pad_max_num=-1, pad_value=0, pad_to_max_length=True):
        
        self.pad_max_num = pad_max_num if pad_max_num != -1 else \
            max([len(ques['node_list']) for ques in self.train_list] \
                + [len(ques['node_list']) for ques in self.val_list])
        self.pad_value = pad_value
        self.pad_to_max_length = pad_to_max_length

    def set_split(self, split="train"):
        assert split in ['train', 'val', 'test']
        self._target_split = split
        self._target_pair, self._target_size = self._lookup_dict[split]

    def __getitem__(self, index):
        QA_item = self._target_pair[index]

        node_list = QA_item['node_list']
        sp_adj = QA_item['sp_adj']

        # no padding
        feature_matrix = torch.cat([n.cls_feature for n in node_list], dim=0)
        adj = torch.from_numpy(sp_adj.to_dense_symmetric())
        
        sent_mask = torch.tensor([1 if n.node_type == 'Sentence' else 0 for n in node_list]).unsqueeze(-1)
        para_mask = torch.tensor([1 if n.node_type == 'Paragraph' else 0 for n in node_list]).unsqueeze(-1)
        
        labels = torch.tensor([1 if n.node_type in ['Paragraph','Sentence'] \
                               and n.is_support else 0 for n in node_list]).unsqueeze(-1)

        answer_type = 1 if node_list[0].answer in ['yes', 'no'] else 0
        ans_yes_no = 1 if node_list[0].answer == 'yes' else 0

        answer_type = torch.tensor([answer_type])
        ans_yes_no = torch.tensor([ans_yes_no])

        # find ans span in top 4 sentences.
        answer_tokens = node_list[0].answer_tokens
        ques_tokens = node_list[0].content_tokens
        sent_tokens = [n.content_tokens for n in node_list if n.node_type == 'Sentence']

        if self.pad_to_max_length:
            node_len = feature_matrix.shape[-2]
            pad_max_num = self.pad_max_num
            pad_value = self.pad_value
            node_dim = feature_matrix.shape[-1]

            feature_matrix_p = torch.zeros([pad_max_num, node_dim]).fill_(pad_value)
            feature_matrix_p[:node_len,:] = feature_matrix[:pad_max_num,:]
            feature_matrix = feature_matrix_p

            adj_p = torch.eye(pad_max_num, pad_max_num)
            adj_p[:node_len,:node_len] = adj[:pad_max_num,:pad_max_num]
            adj = adj_p

            sent_mask_p = torch.zeros([pad_max_num, 1]).fill_(pad_value)
            sent_mask_p[:node_len,:] = sent_mask[:pad_max_num,:]
            sent_mask = sent_mask_p

            para_mask_p = torch.zeros([pad_max_num, 1]).fill_(pad_value)
            para_mask_p[:node_len,:] = para_mask[:pad_max_num,:]
            para_mask = para_mask_p
            
            labels_p = torch.zeros([pad_max_num, 1]).fill_(pad_value)
            labels_p[:node_len,:] = labels[:pad_max_num,:]
            labels = labels_p
            
        assert not torch.any(torch.isnan(feature_matrix)).item()

        item_info_dict = {
            'feature_matrix': feature_matrix,
            'adj': adj,
            'sent_mask': sent_mask,
            'para_mask': para_mask,
            'labels': labels,
            'answer_type': answer_type,
            'ans_yes_no': ans_yes_no,
            'ques_tokens': ques_tokens,
            'answer_tokens': answer_tokens,
            'sent_tokens': sent_tokens,
        }

        return item_info_dict

    def __len__(self):
        return self._target_size
    
    def __repr__(self):
        return 'HotpotQA GNN Dataset. mode: {}. size: {}. max_seq: {}'.format\
            (self._target_split, self.__len__(), self.pad_max_num)

    def __str__(self):
        return self.__repr__()

    def get_num_batches(self, batch_size):
        return len(self) // batch_size

def gen_GNN_batches(dataset, batch_size, shuffle=True, drop_last=True, device='cpu', seed = 123):

    if seed: np.random.seed(seed)
    dataset_len = dataset.__len__()
    
    index_pool = [i for i in range(dataset_len)]
    if shuffle: np.random.shuffle(index_pool)

    cursor = 0
    while cursor < dataset_len:
        # last batch
        if cursor + batch_size > dataset_len:
            if drop_last: break

        FLAG_FIRST = True
        for index in index_pool[cursor: min(cursor + batch_size, dataset_len)]:
            res_dict = dataset[index]
            if FLAG_FIRST:
                feature_matrix = res_dict['feature_matrix'].unsqueeze(0)
                adj = res_dict['adj'].unsqueeze(0)
                
                sent_mask = res_dict['sent_mask'].unsqueeze(0)
                para_mask = res_dict['para_mask'].unsqueeze(0)
                labels = res_dict['labels'].unsqueeze(0)

                answer_type = res_dict['answer_type']
                ans_yes_no = res_dict['ans_yes_no']
                
                ques_tokens = res_dict['ques_tokens']
                answer_tokens = res_dict['answer_tokens']
                sent_tokens = []
                
                FLAG_FIRST = False
            else:
                feature_matrix = torch.cat([feature_matrix, res_dict['feature_matrix'].unsqueeze(0)], dim=0)
                adj = torch.cat([adj, res_dict['adj'].unsqueeze(0)], dim=0)
                
                sent_mask = torch.cat([sent_mask, res_dict['sent_mask'].unsqueeze(0)], dim=0)
                para_mask = torch.cat([para_mask ,res_dict['para_mask'].unsqueeze(0)], dim=0)
                labels = torch.cat([labels, res_dict['labels'].unsqueeze(0)], dim=0)
                
                answer_type = torch.cat([answer_type, res_dict['answer_type']], dim=-1)
                ans_yes_no = torch.cat([ans_yes_no, res_dict['ans_yes_no']], dim=-1)

            sent_tokens.append(res_dict['sent_tokens'])

        cursor += batch_size
        
        feature_matrix = feature_matrix.to(device)
        adj = adj.long().to(device)
        sent_mask = sent_mask.long().to(device)
        para_mask = para_mask.long().to(device)
        labels = labels.long().to(device)
        answer_type = answer_type.long().to(device)
        ans_yes_no = ans_yes_no.long().to(device)
        
        batch_item_info_dict = {
            'feature_matrix': feature_matrix,
            'adj': adj,
            'sent_mask': sent_mask,
            'para_mask': para_mask,
            'labels': labels,
            'answer_type': answer_type,
            
            # model2 
            'ans_yes_no': ans_yes_no,
            'ques_tokens':ques_tokens,
            'answer_tokens': answer_tokens,
            'sent_tokens': sent_tokens,
        }
        for k,v in batch_item_info_dict.items():
            assert type(v)==list or not torch.isnan(v).any()

        yield batch_item_info_dict

class HotpotQA_QA_Dataset(Dataset):
    def __init__(self, train_list, val_list):
  
        # elements
        self.train_list = train_list # json files path list.
        self.train_size = len(self.train_list)

        self.val_list = val_list
        self.val_size = len(self.val_list)

        # func
        self._lookup_dict = {'train': (self.train_list, self.train_size),
                            'val': (self.val_list, self.val_size)}
        self.set_split('train')
        
        # parameters
        self.set_parameters()

    @classmethod
    def build_dataset(cls, json_path = 'data/hotpot_train_mini.json', ratio_train=0.7, seed=123):

        with open(json_path, 'r', encoding='utf-8') as fp:
            json_file = json.load(fp)
                            
        np.random.seed(seed)
        np.random.shuffle(json_file)        
        
        sample_train_num = int(ratio_train * len(json_file))

        train_list = json_file[:sample_train_num]
        val_list = json_file[sample_train_num:]

        return cls(train_list, val_list)
        
    def set_parameters(self, tokenizer=None, topN_sents=4, max_length=512, uncased=True, permutations=False, random_seed=123):
        """uncased: convert to lower chars. """
        self.tokenizer = tokenizer
        self.topN_sents = topN_sents
        self.max_length = max_length
        self.random_seed = random_seed
        self.permutations = permutations
        self.uncased = uncased
        
    def set_split(self, split="train"):
        assert split in ['train', 'val', 'test']
        self._target_split = split
        self._target_pair, self._target_size = self._lookup_dict[split]
    
    def clean_text(self, text):
        # clean number.
        text = re.sub(r'[0-9]{1,3}(,)[0-9]{3}',
                                lambda x: f" {x.group(0).replace(',','')} ",
                                text)
        # clean punctuations.
        text = re.sub(r'[^\w\s]',' ',text)
        text = re.sub(r' +',' ',text)

        # lower
        if self.uncased:
            text = text.lower()
        return text

    def _remove_quesid(self, inputs_id):
        if 'XLNetTokenizer' in str(self.tokenizer.__class__):
            SEP = self.tokenizer.sep_token_id
            for end in range(len(inputs_id)):
                if inputs_id[end] != SEP: continue
                inputs_id[0:end] = [-99 for _ in range(end)]
                break
            return inputs_id
        else:
            if str(type(self.tokenizer)).split('.')[-1].split('\'')[0] in ['DistilBertTokenizer', 'BertTokenizer']:
                BOS = 101
                EOS = 102
            else:
                BOS = self.tokenizer.bos_token_id
                EOS = self.tokenizer.eos_token_id
            for start in range(len(inputs_id)):
                if inputs_id[start] != BOS: continue
                for end in range(start+1, len(inputs_id)):
                    if inputs_id[end] != EOS: continue
                    inputs_id[start+1:end] = [-99 for _ in range(end - start - 1)]
                    break
                break
            return inputs_id

    def __getitem__(self, index, check_err_mod=False, detail_mod=False):
        item = self._target_pair[index]
        support_sents = []
        
        for supports in item['supporting_facts']:
            for para in item['context']:
                if para[0] == supports[0]:
                    try:
                        # concat paragraph title.
                        support_sents.append(f"{para[0]} : {para[1][supports[1]]}")
                    except IndexError:
                        if check_err_mod: print(f"{self._target_split} {index}. {item['_id']} index err.")
                        pass

        all_other_sents = []
        for para in item['context']:
            if para[1] not in support_sents: all_other_sents.extend(para[1])

        if len(support_sents) < self.topN_sents:
            np.random.seed(self.random_seed)
            try:
                support_sents.extend(np.random.choice(all_other_sents, 
                                                      size=self.topN_sents - len(support_sents),
                                                      replace=False))
            except ValueError:
                # support sentences are less than N.
                support_sents.extend(np.random.choice(all_other_sents, 
                                                      size=self.topN_sents - len(support_sents),
                                                      replace=True))
        else:
            support_sents = support_sents[:self.topN_sents]
        
        support_sents = [self.clean_text(i) for i in support_sents]
        if self.permutations:
            permutated_sents = [' '.join(i) for i in permutations(support_sents, len(support_sents))]
            np.random.shuffle(permutated_sents)
            support_sents = permutated_sents
        else:
            support_sents = [' '.join(support_sents)]

        item['question'] = self.clean_text(item['question'])
        item['answer'] = self.clean_text(item['answer'])

        ques_sents_list = [[(item['question'], sent)] for sent in support_sents]
        ans_ids = self.tokenizer.encode(item['answer'], add_special_tokens=True)

        if 'XLNetTokenizer' in str(self.tokenizer.__class__):
            ans_ids = ans_ids[:-2]
        else:
            ans_ids = ans_ids[1:-1]

        input_ids_list = []
        token_type_ids_list = []
        attention_mask_list = []
        special_tokens_mask_list = []
        start_positions_list = []
        end_positions_list = []
        yes_no_span_list = []

        for ques_sents in ques_sents_list:
            res = self.tokenizer.batch_encode_plus(ques_sents, 
                                                    max_length=self.max_length,
                                                    pad_to_max_length=True, 
                                                    return_tensors='pt',
                                                    return_special_tokens_masks=True)
            # RoBERTa does not make use of token type ids
            if 'token_type_ids' not in res:
                res['token_type_ids'] = torch.zeros_like(res['input_ids'])
            token_type_ids_list.append(res['token_type_ids'])
            attention_mask_list.append(res['attention_mask'])
            special_tokens_mask_list.append(res['special_tokens_mask'])
            input_ids_list.append(res['input_ids'][0])

            if item['answer'] == 'yes':
                yes_no_span = [1]
                start_positions = [-100]
                end_positions = [-100]
            elif item['answer'] == 'no':
                yes_no_span = [0]
                start_positions = [-100]
                end_positions = [-100]
            else:
                yes_no_span = [-100]  
                input_remove_quesid = self._remove_quesid(res['input_ids'][0].tolist())
                start_end_list = find_ans_spans(ans_ids, input_remove_quesid, top_num = 1)
                if start_end_list == []:
                    if check_err_mod: 
                        print(f"{self._target_split} {index}. {item['_id']} no answer: {item['answer']}\tsupport sent: {len(item['supporting_facts'])}")
                    start_positions = [-100]
                    end_positions = [-100]
                else:
                    start_positions = [start_end_list[0][0]]
                    end_positions = [start_end_list[0][1]]
            
            if detail_mod:
                print(item['question'])
                print(ques_sents)
                print(item['answer'])
                print(ans_ids)
                print('input_remove_quesid')
                print(input_remove_quesid)

            start_positions_list.append(torch.tensor(start_positions))
            end_positions_list.append(torch.tensor(end_positions))
            yes_no_span_list.append(torch.tensor(yes_no_span))

        # concate permutations. shape 1 should be permutation dimantion.
        item_info_dict = {
            'input_ids': torch.stack(input_ids_list).long(),
            'token_type_ids': torch.stack(token_type_ids_list).long(),
            'attention_mask': torch.stack(attention_mask_list).long(),
            'special_tokens_mask': torch.stack(special_tokens_mask_list).long(),
            'start_positions': torch.stack(start_positions_list).long(),
            'end_positions': torch.stack(end_positions_list).long(),
            'yes_no_span': torch.stack(yes_no_span_list).long(),
            }

        return item_info_dict

    def get_item_err_detail_by_id(self, _id):
        for mode in ['train', 'val']:
            self.set_split(mode)
            for index in range(self._target_size):
                if self._target_pair[index]['_id'] == _id:
                    self.__getitem__(index=index, detail_mod = True)
                    return 
            
    def check_all(self):
        '''check index error and no-answer error.'''
        for mode in ['train', 'val']:
            self.set_split(mode)
            for index in range(self._target_size):
                self.__getitem__(index=index, check_err_mod = True)
    
    def check_item(self, _id):
        for item in self.train_list + self.val_list:
            if item['_id'] == _id:
                print(item)

    def check_supporting_facts(self, _id):
        for item in self.train_list + self.val_list:
            if item['_id'] == _id:
                print(f"supporting_facts: {item['supporting_facts']}")        
                print(f"answer: {item['answer']}") 
                print('')
                for support in item['supporting_facts']:
                    for para in item['context']:
                        if para[0] == support[0]:
                            print(para[0])
                            print(para[1][support[1]])

    def __len__(self):
        return self._target_size
    
    def __repr__(self):
        return 'HotpotQA QA Dataset. mode: {}. size: {}. sents num: {}'.format\
            (self._target_split, self.__len__(), self.topN_sents)

    def __str__(self):
        return self.__repr__()

    def get_num_batches(self, batch_size):
        return len(self) // batch_size

def find_ans_spans(target, tokens, offets_type = 'position', top_num = None):
    assert offets_type in ['position', 'range'] and \
        type(target) == type(tokens) == list
    len_x1 = len(target)
    len_x2 = len(tokens)
    if len_x1 == 0 or len_x2 == 0 or len_x1 > len_x2:
        return []
    
    i1=0
    i2=0
    i2_current = 0
    spans = []
    while i2 <= len_x2 - len_x1:
        if top_num and len(spans) == top_num:
            break
        i2_current = i2
        while i1 < len_x1:
            if target[i1] != tokens[i2]: 
                i1 = 0
                i2 = i2_current + 1
                break
            else:
                i1 += 1
                i2 += 1
                
        if not i1 < len_x1:
            i1 = 0
            if offets_type == 'position':  
                spans.append([i2_current, i2_current+len_x1-1])
            else:
                spans.append([i2_current, i2_current+len_x1])
                
    return spans[:top_num+1] if top_num else spans[:]

def generate_QA_batches(dataset, batch_size, shuffle=True, drop_last=True, device="cpu"): 

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            if name in ['start_positions', 'end_positions', 'yes_no_span']:
                out_data_dict[name] = data_dict[name].view(-1)
            else:
                # do it for permutations.
                last_size = tensor.shape[-1] # seq len.
                out_data_dict[name] = data_dict[name].view(-1, last_size)

        yield out_data_dict






