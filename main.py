from Classes import *

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
    question_tokens = tokensizer_in_Model(question)
    Q_node = Node.build(Q_node_cursor, 'Question', question, -1)
    node_list.append(Q_node)

    ques_para_list = []
    
    for p_id, paragraph in enumerate(context):
        # Q-P content
        para_id = f'{Q_id}_{p_id}'
        paragraph_tokens = []
        paragraph_offset = 0
        paragraph_label = 0

        title = paragraph[0]
        title_tokens = tokensizer_in_Model(title)
        
        print(title_tokens)

        paragraph_tokens += title_tokens
        paragraph_offset += len(title_tokens)

        index_cursor += 1

        # 添加P-1和P
        try:
            if index_cursor != 0: sp_adj.append(edge_type_map['P_P'], P_node_cursor, index_cursor) 
        except (NameError): # 首次调用P_node_cursor会报错
            pass
        P_node_cursor = index_cursor
        P_node = Node.build(P_node_cursor, 'Paragraph', title, Q_node_cursor)
        P_node.set_span_in_paragraph(para_id, paragraph_offset - len(title_tokens), paragraph_offset)
        # 判断 support paragraph
        if title in supporting_facts.keys():
            paragraph_label = 1
            P_node.set_support()
        node_list.append(P_node)
        
        # 添加Q和P
        sp_adj.append(edge_type_map['Q_P'], Q_node_cursor, P_node_cursor)

        for s_index, sentence in enumerate(paragraph[1]):
            sentence_tokens = tokensizer_in_Model(sentence)
            paragraph_tokens += sentence_tokens
            paragraph_offset += len(sentence_tokens)

            index_cursor += 1
            S_node_cursor = index_cursor
            S_node = Node.build(S_node_cursor, 'Sentence', sentence, P_node_cursor)
            S_node.set_span_in_paragraph(para_id, paragraph_offset - len(sentence_tokens), paragraph_offset)

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
                sentence_offset = paragraph_offset - len(sentence_tokens)

                index_cursor += 1
                E_node_cursor = index_cursor
                E_node = Node.build(E_node_cursor, 'Entity', entities_dict['content'], S_node_cursor)
                E_node.set_span_in_paragraph(para_id,\
                                sentence_offset+entities_dict['span_start'],\
                                sentence_offset+entities_dict['span_end'])

                node_list.append(E_node)
                sp_adj.append(edge_type_map['S_E'], S_node_cursor, E_node_cursor)

        # in paragraph loop.
        ques_para = Question_Paragraph.build(question, paragraph_tokens, Q_id, para_id, paragraph_label)
        ques_para_list.append(ques_para)
    
    # in item loop
    # 连接Q节点和E节点.
    E_nodes_in_Q = [i for i in node_list if i.node_type == 'Entity' and i.content_raw.replace(' ','') in question.replace(' ','')]
    for i in E_nodes_in_Q:
        sp_adj.append(edge_type_map['Q_E'], Q_node_cursor, i.node_id)
    
    # 连接S和P节点
    # 注意!!原文没有使用entity linking系统, 而是直接使用Wikipedia提供的链接.
    # 这里暂时使用字符串match.
    S_node = [i for i in node_list if i.node_type == 'Sentence']
    E_nodes = [i for i in node_list if i.node_type == 'Entity']
    for E_n in E_nodes:
        entity = E_n.content_raw.replace(' ','')
        for S_n in S_node:
            if entity in S_n.content_raw.replace(' ',''):
                sp_adj.append(edge_type_map['S_S_hyper'], node_list[E_n.parent_id].node_id, node_list[S_n.parent_id].node_id)
            
    
    ques_info_dict['node_list'] = node_list
    ques_info_dict['sp_adj'] = sp_adj
    ques_info_dict['ques_para_list'] = ques_para_list
    return ques_info_dict

from tqdm import tqdm
from multiprocessing import Pool
def preprocessing(item_num = 2, thread_num = 1):
    '''main multi-processes function.'''
    item_num = None if item_num<0 else item_num
    thread_num = 1 if thread_num<0 else thread_num
    
    resturn_list = []
    pbar = tqdm(total = len(json_train[:item_num]), desc = f'processing json items')
    with Pool(thread_num) as pool:
        pool_iter = pool.imap(_process, json_train[:item_num])
        for i,r in enumerate(pool_iter):
            resturn_list.append(r)
            pbar.update()
    return resturn_list


if __name__ == "__main__":
    try:
        # from google.colab import drive
        # drive.mount('/content/folders/')
        # !pip install transformers
        # !pip install -U spacy[cuda100]
        # !wget -P /content/folders/My\ Drive/download/ https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-2.2.5/en_core_web_lg-2.2.5.tar.gz
        # !pip install /content/folders/My\ Drive/download/en_core_web_lg-2.2.5.tar.gz
        # !wget -P /content/folders/My\ Drive/HotpotQA/ http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
        # json_train_path = '/content/folders/My Drive/HotpotQA/样例_hotpot_train_v1.1.json' # 5个例子
        json_train_path = '/content/folders/My Drive/HotpotQA/hotpot_train_v1.1.json'
        save_cache_path = '/content/folders/My Drive/save_cache/'
        save_cache_path_linux = '/content/folders/My\ Drive/save_cache/'
    except:
        json_train_path = r'D:\github_work\TBQA\TBQA_data\HotpotQA\example_hotpot_train_v1.1.json'

    
    with open(json_train_path, 'r', encoding='utf-8') as fp:
        json_train = json.load(fp)
    
    # print(json_train)
    hotpotQA_train_preprocess = preprocessing(10,2)
    print (hotpotQA_train_preprocess)




