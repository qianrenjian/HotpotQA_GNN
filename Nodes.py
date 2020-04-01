import torch
import numpy as np
import scipy.sparse as sp

class BaseNode(object):
    '''Node class for graph'''
    
    def __init__(self, node_id, node_type, parent_id,
                 content_raw, content_tokens=None, cls_feature=None):
        
        self.node_id = node_id
        self.node_type = node_type
        self.parent_id = parent_id

        self.content_raw = content_raw
        self.content_tokens = content_tokens

        self.cls_feature = cls_feature # final features. [1,dim]
        
    @classmethod
    def build(cls):
        raise NotImplementedError
        
    def to_serializable(self):
        raise NotImplementedError

    @classmethod
    def from_serializable(cls, contents):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError
    
    def __repr__(self):
        raise NotImplementedError

class QuestionNode(BaseNode):
    def __init__(self,node_id, node_type, parent_id, content_raw, content_tokens,
                 answer, answer_tokens, ques_type, cls_feature=None):
        super(QuestionNode, self).__init__(node_id, node_type, parent_id, 
                                           content_raw,content_tokens,cls_feature)

        self.question = self.content_raw
        self.answer = answer
        self.answer_tokens = answer_tokens
        self.ques_type = ques_type

    @classmethod
    def build(cls, node_id, parent_id, content_raw,
                answer, ques_type):
        
        node_type = 'Question'
        answer_tokens = tokenizer.tokenize(answer)
        content_tokens = tokenizer.tokenize(content_raw)
        return cls(node_id, node_type, parent_id, content_raw, content_tokens,\
                    answer, answer_tokens, ques_type)

    def to_serializable(self):
        return {
            'node_id': self.node_id,
            'node_type': self.node_type,
            'parent_id': self.parent_id,
            'content_raw': self.content_raw,
            'content_tokens': self.content_tokens,
            'answer': self.answer,
            'answer_tokens': self.answer_tokens,
            'ques_type': self.ques_type,
            'cls_feature': self.cls_feature.tolist(),
               }

    @classmethod
    def from_serializable(cls, state_dicts):
        state_dicts['cls_feature'] = torch.tensor(state_dicts['cls_feature'])
        return cls(**state_dicts)
    
    def __str__(self):
        return f'QuestionNode: {self.node_id}'
    
    def __repr__(self):
        return self.__str__()

class ParagraphTitleNode(BaseNode):
    '''不含整个段落, 只含有title'''
    def __init__(self,node_id, node_type, parent_id, content_raw, content_tokens,
                 content_NER_list = None, cls_feature=None, is_support=False):
        super(ParagraphTitleNode, self).__init__(node_id, node_type, parent_id, 
                                           content_raw,content_tokens,cls_feature)

        self.content_NER_list = content_NER_list

        self.is_support = is_support # 段落 句子 

    @classmethod
    def build(cls, node_id, parent_id, content_raw):

        _, content_NER_list = find_NER_in_spacy(content_raw, ner=True)
        content_tokens = tokenizer.tokenize(content_raw)
        #  content_NER_list = find_NER_in_Model(content_raw, content_tokens)
        # content_tokens: NO CLS.

        node_type = 'Paragraph'
        return cls(node_id, node_type, parent_id, content_raw, content_tokens,\
                    content_NER_list)
    
    def set_support(self):
        self.is_support = True

    def to_serializable(self):
        return {
            'node_id': self.node_id,
            'node_type': self.node_type,
            'parent_id': self.parent_id,
            'content_raw': self.content_raw,
            'content_tokens': self.content_tokens,
            'content_NER_list': self.content_NER_list,
            'cls_feature': self.cls_feature.tolist(),
            'is_support': self.is_support,
               }

    @classmethod
    def from_serializable(cls, state_dicts):
        state_dicts['cls_feature'] = torch.tensor(state_dicts['cls_feature'])
        return cls(**state_dicts)
        
    def __str__(self):
        return f'ParagraphTitleNode: {self.node_id}'
    
    def __repr__(self):
        return self.__str__()

class SentenceNode(BaseNode):
    def __init__(self,node_id, node_type, parent_id, content_raw, content_tokens,\
                    content_NER_list = None, cls_feature=None, is_support = False):
        super(SentenceNode, self).__init__(node_id, node_type, parent_id, 
                                           content_raw,content_tokens,cls_feature)

        self.content_NER_list = content_NER_list

        self.is_support = is_support # 段落 句子 

    @classmethod
    def build(cls, node_id, parent_id, content_raw):

        _, content_NER_list = find_NER_in_spacy(content_raw, ner=True)
        content_tokens = tokenizer.tokenize(content_raw)
        # content_NER_list = find_NER_in_Model(content_raw, content_tokens)
        # content_tokens: NO CLS.

        node_type = 'Sentence'
        return cls(node_id, node_type, parent_id, content_raw, content_tokens,\
                    content_NER_list)

    def set_support(self):
        self.is_support = True

    def get_NER_tuples_list(self):
        '''返回NER元组. e.g. [('ALLPE',id), ('DELL',id)]'''
        return [(i['content'], self.node_id) for i in self.content_NER_list]

    def to_serializable(self):
        return {
            'node_id': self.node_id,
            'node_type': self.node_type,
            'parent_id': self.parent_id,
            'content_raw': self.content_raw,
            'content_tokens': self.content_tokens,
            'content_NER_list': self.content_NER_list,
            'cls_feature': self.cls_feature.tolist(),
            'is_support': self.is_support,
               }

    @classmethod
    def from_serializable(cls, state_dicts):
        state_dicts['cls_feature'] = torch.tensor(state_dicts['cls_feature'])
        return cls(**state_dicts)
    
    def __str__(self):
        return f'SentenceNode: {self.node_id}'
    
    def __repr__(self):
        return self.__str__()

class EntityNode(BaseNode):
    def __init__(self,node_id, node_type, parent_id, content_raw, 
                 content_tokens, cls_feature=None):
        super(EntityNode, self).__init__(node_id, node_type, parent_id, 
                                           content_raw,content_tokens,cls_feature)

    @classmethod
    def build(cls, node_id, parent_id, content_raw):

        node_type = 'Entity'
        content_tokens = tokenizer.tokenize(content_raw)
        return cls(node_id, node_type, parent_id, content_raw,content_tokens)
    
    def to_serializable(self):
        return {
            'node_id': self.node_id,
            'node_type': self.node_type,
            'parent_id': self.parent_id,
            'content_raw': self.content_raw,
            'content_tokens': self.content_tokens,
            'cls_feature': self.cls_feature.tolist(),
               }

    @classmethod
    def from_serializable(cls, state_dicts):
        state_dicts['cls_feature'] = torch.tensor(state_dicts['cls_feature'])
        return cls(**state_dicts)
        
    def __str__(self):
        return f'EntityNode: {self.node_id}'
    
    def __repr__(self):
        return self.__str__()

class Adjacency_sp():
    '''无重复稀疏邻接矩阵'''
    def __init__(self, v_i_j=[]):
        self.v_i_j = v_i_j
        self.i_j_find_table = []

    def append(self, v, i, j):
        if not (i,j) in self.i_j_find_table:
            self.v_i_j.append([v,i,j])
            self.i_j_find_table.append((i,j))
    
    def to_dense(self):
        '''return numpy ndarray.'''
        np_adj = np.array(self.v_i_j)
        node_len = max(max(np_adj[:, 1]), max(np_adj[:, 2])) + 1
        full_adj = sp.coo_matrix((np_adj[:, 0], (np_adj[:, 1], np_adj[:, 2])), 
                                 shape=(node_len,node_len), dtype=np.float32).todense()
        full_adj = np.array(full_adj) + np.eye(node_len,node_len)
        return full_adj

    def to_dense_symmetric(self):
        '''self-loop symmetric adj matrix.'''
        np_adj = np.array(self.v_i_j)
        node_len = max(max(np_adj[:, 1]), max(np_adj[:, 2])) + 1
        adj = sp.coo_matrix((np_adj[:, 0], (np_adj[:, 1], np_adj[:, 2])), 
                            shape=(node_len,node_len), dtype=np.float32)
        adj_symm = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj).todense()
        adj_symm = np.array(adj_symm) + np.eye(node_len,node_len)
        return adj_symm

    def to_serializable(self):
        return {
            'v_i_j': self.v_i_j,
               }

    @classmethod
    def from_serializable(cls, state_dicts):
        return cls(**state_dicts)
    
    def __repr__(self):
        np_adj = np.array(self.v_i_j)
        return f'Adjacency_sp has {len(self.v_i_j)} edges {max(max(np_adj[:, 1]), max(np_adj[:, 2])) + 1} nodes.'
    def __str__(self):
        return self.__repr__()
    def __len__(self):
        return len(self.v_i_j)

def auto_reload_Node(state_dicts):
    node_type = state_dicts['node_type']
    if node_type == 'Question':
        return QuestionNode.from_serializable(state_dicts)
    elif node_type == 'Paragraph':
        return ParagraphTitleNode.from_serializable(state_dicts)
    elif node_type == 'Sentence':
        return SentenceNode.from_serializable(state_dicts)
    elif node_type == 'Entity':
        return EntityNode.from_serializable(state_dicts)
    else:
        raise
