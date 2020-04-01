import torch

# 辅助函数
def get_feature_from_model(text, text_pair=None, return_type = 'first', \
                           max_length=512, tokenizer=None, LM_model=None):
    '''输入[seq_len], 返回[seq_len, dim]'''
    assert return_type in ['first','second']
    assert LM_model.config.output_hidden_states == True
    
    if text_pair == None: return_type = 'first'
    
    model_input = tokenizer.encode_plus(
            text = text,
            text_pair = text_pair,
            add_special_tokens = True,
            max_length = max_length,
            truncation_strategy = 'only_second',
            pad_to_max_length = False,
            return_tensors = 'pt'
        )

    res_tuple=LM_model(**model_input)
    first_part_len = model_input['token_type_ids'].flatten().tolist().count(0)
    
    # last hidden layer.
    seq_hidden = res_tuple[1][-1].squeeze()
    
    if text_pair == None or return_type == 'first': 
        return seq_hidden[1:-1]
    else:
        # first_part_len-1 is [SEP]
        return seq_hidden[first_part_len:-1]

class Node(object):
    '''Node class for graph'''
    
    def __init__(self, node_id, node_type, content_raw, content_tokens, \
                 content_NER_list, parent_id, content_features=None, is_support=False):
        self.node_id = node_id
        self.node_type = node_type
        self.content_raw = content_raw
        self.content_tokens = content_tokens
        self.content_NER_list = content_NER_list
        self.parent_id = parent_id
        
        # Q_node doesn't have.
        self.paragraph_id = -1
        self.start_in_paragraph = -1
        self.end_in_paragraph = -1

        # only for E_node
        # E节点能够通过parent_id找到S节点.
        self.start_in_sentence = -1
        self.end_in_sentence = -1

        self.is_support = is_support # 段落 句子 

        self.content_features = content_features
        self.cls_feature = None # final features. [1,dim]
        
    @classmethod
    def build(cls, node_id, node_type, content_raw, parent_id, content_tokens=None):
        '''content_tokens能加快计算速度.'''
        # content_tokens = tokensize_and_repr_in_BERT(content_raw, flatten=True)
        if node_type != 'Entity':
            content_tokens_NOCLS, content_NER_list = find_NER_in_Model(content_raw, content_tokens)
        else:
            content_tokens_NOCLS = content_tokens
            content_NER_list = None
        # print(f'id:{node_id}\n{content_raw}\n{content_NER_list}\n')
        return cls(node_id, node_type, content_raw, content_tokens_NOCLS, content_NER_list, parent_id)
    
    def set_support(self):
        self.is_support = True
    
    def set_span_in_paragraph(self, para_id, start):
        self.paragraph_id = para_id
        self.start_in_paragraph = start
        self.end_in_paragraph = start + len(self.content_tokens)

    # only for E_node.
    def set_span_in_sentence(self, start):
        self.start_in_sentence = start
        self.end_in_sentence = start + len(self.content_tokens)
        
    def __str__(self):
        return f'Node: {self.node_type} {self.node_id}'
    
    def __repr__(self):
        return f'Node: {self.node_type} {self.node_id}'

    def get_NER_tuples_list(self):
        '''返回NER元组. e.g. [('ALLPE',id), ('DELL',id)]'''
        return [(i['content'], self.node_id) for i in self.content_NER_list]

class Question_Paragraph(object):
    '''Q-P pair and label. for BERT and node init.
    返回q-p对和q-s对, 还要确保能够初始化Node类.
    每个问句有10个paragraph,即10个此类.'''
    def __init__(self, ques_id, para_id, question_tokens, para_title_tokens, para_label, sents_in_para, sentences_label):
        self.question_tokens = question_tokens
        self.para_title_tokens = para_title_tokens
        self.sents_in_para = sents_in_para
        self.sentences_offsets = self.cal_offsets(self.sents_in_para)
        self.sentences_label = sentences_label

        self.ques_id = ques_id
        self.para_id = para_id 
        self.para_label = para_label # 段落label

        self.question_features = None # [N, dim]
        self.para_features = None
#         self.paragraph_features = None

    @classmethod
    def build(cls, ques_id, para_id, question, para_title_tokens, para_label, node_list):
        Snodes = [n for n in node_list if n.paragraph_id == para_id and n.node_type == 'Sentence']
        question_tokens = tokensizer_in_Model(question)
        # para_title_tokens = tokensizer_in_Model(para_title)
        sents_in_para = [n.content_tokens for n in Snodes]
        sentences_label = [int(n.is_support) for n in Snodes]
        return cls(ques_id, para_id, question_tokens, para_title_tokens, para_label, sents_in_para, sentences_label)

    @staticmethod
    def cal_offsets(sents_list):
        cursor = 0
        offsets = []
        for sent_tokens in sents_list:
            len_sent = len(sent_tokens)
            offsets.append((cursor, cursor+len_sent))
            cursor += len_sent
        return offsets
    
    # content tokens
    def get_para_tokens(self, contain_title = False):
        para_token = self.para_title_tokens if contain_title else []
        for i in self.sents_in_para: para_token.extend(i)
        return para_token

    def get_ques_para_label_tuple(self, contain_title = False):
        '''问句-段落对'''
        return (self.question_tokens, self.get_para_tokens(contain_title), self.para_label)

    def get_ques_sent_label_list(self, contain_title = False):
        '''问句-句子对'''
        if contain_title:
            return [(self.question_tokens, self.para_title_tokens.extend(sent_tokens), sent_label)\
                for sent_tokens,sent_label in zip(self.sents_in_para, self.sentences_label)]
        else:
            return [(self.question_tokens, sent_tokens, sent_label)\
                for sent_tokens,sent_label in zip(self.sents_in_para, self.sentences_label)]

    def format_sents_in_para(self):
        return ' '.join([f'{index}:{word}' for index,word in enumerate(self.sents_in_para)])

    # features
    def build_features(self):
        '''build features from LM models'''
    
        self.question_features = get_feature_from_model(self.question_tokens)

        para_features = 0
        for one_line in self.get_ques_sent_label_list():

            sent_features = get_feature_from_model(one_line[0], one_line[1], 'second')

            if type(para_features) == int: para_features = sent_features.clone()
            else: para_features = torch.cat((para_features, sent_features), dim=0)
       
        self.para_features = para_features
        

    def get_question_features(self):
        return self.question_features

    def get_paragraph_features(self):
        return self.para_features

    # other
    def __str__(self):
        return f'Q_P. p_id: {self.para_id}'
    
    def __repr__(self):
        return f'Q_P. p_id: {self.para_id}'

import scipy.sparse as sp
import numpy as np

class Adjacency_sp(object):
    '''无重复稀疏邻接矩阵'''
    def __init__(self):
        self.v_i_j = []
        self.i_j_find_table = []

    def append(self, v, i, j):
        if not (i,j) in self.i_j_find_table:
            self.v_i_j.append([v,i,j])
            self.i_j_find_table.append((i,j))
    
    def to_dense(self):
        '''return numpy ndarray.'''
        _len = max([i[0] for i in self.i_j_find_table] + [i[1] for i in self.i_j_find_table]) + 1
        shape = (_len,_len)
        np_adj = np.array(self.v_i_j)
        full_adj = sp.coo_matrix((np_adj[:, 0], (np_adj[:, 1], np_adj[:, 2])), shape=shape, dtype=np.float32).todense()
        full_adj = np.array(full_adj)
        return full_adj

    def to_dense_symmetric(self):
        _len = max([i[0] for i in self.i_j_find_table] + [i[1] for i in self.i_j_find_table]) + 1
        shape = (_len,_len)
        np_adj = np.array(self.v_i_j)
        adj = sp.coo_matrix((np_adj[:, 0], (np_adj[:, 1], np_adj[:, 2])), shape=shape, dtype=np.float32)
        adj_symm = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj).todense()
        adj_symm = np.array(adj_symm)
        return adj_symm

    def __repr__(self):
        return f'Adjacency_sp has {len(self.v_i_j)} edges'
    def __str__(self):
        return f'Adjacency_sp has {len(self.v_i_j)} edges'
    def __len__(self):
        return len(self.v_i_j)