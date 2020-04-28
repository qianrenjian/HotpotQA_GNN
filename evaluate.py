from argparse import Namespace
import argparse
import json
from collections import defaultdict
import time
from tqdm import tqdm

import torch
from transformers import AutoTokenizer

from GNN import GAT_HotpotQA
from gen_nodes_repr import build_for_one_item
from datasets import HotpotQA_GNN_Dataset, HotpotQA_QA_Dataset, gen_GNN_batches, generate_QA_batches
from QA_models import AutoQuestionAnswering


# args = Namespace(
#     # Data and model path.
#     dev_json_path = 'data/HotpotQA/hotpot_dev_distractor_v1.json',
#     GNN_model_path = 'save_model_GNN/GNN_HotpotQA_hidden64_heads8_pad300_chunk_first.pt',
#     QA_model_path = 'save_model_QA_permutations/HotpotQA_QA_BiGRU_roberta-base-squad2.pt',
#     model_path = 'data/models/roberta-base-squad2',

#     # GNN parameters. MUST match saved pt file.
#     features = 768,
#     hidden = 64,
#     nclass = 2,
#     dropout = 0,
#     alpha = 0.3,
#     nheads = 8,
#     pad_max_num = 300,

#     # Parameters for QA.
#     header_mode='MLP',
#     cls_token_id = 0,
#     topN_sents = 3,
#     max_length = 512,
#     uncased = False,

#     # Runtime hyper parameter
#     cuda=True,
#     device=None,
#     )

def set_envs(args):
    if not args.device:
        args.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if not args.cuda:
        args.device = torch.device("cpu")

def eval(args, dev_json):
    test_nums = args.test_nums if args.test_nums >=0 else None
    ques_items = build_for_one_item(dev_json[:test_nums], args)
    classifierGNN = GAT_HotpotQA(features=args.features, hidden=args.hidden, nclass=args.nclass, 
                                dropout=args.dropout, alpha=args.alpha, nheads=args.nheads, 
                                nodes_num=args.pad_max_num)
    checkpoint = torch.load(args.GNN_model_path)
    try:
        classifierGNN.load_state_dict(checkpoint['model'])
    except:
        classifierGNN.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['model'].items()})
    _ = classifierGNN.eval()
    classifierGNN.to(args.device)

    datasetGNN = HotpotQA_GNN_Dataset.load_for_eval(ques_items)
    datasetGNN.set_parameters(300,0)
    print(datasetGNN)

    # GNN eval.
    pbar_GNN = tqdm(total = datasetGNN.get_num_batches(1), desc="GNN EVAL")
    batch_generator = gen_GNN_batches(datasetGNN, 1, shuffle=False, drop_last=False, device=args.device)
    sup_dict = {}
    sup_raw_dict = {}
    QA_eval_list, Qtype_list = [], [] # for model 2.
    for index, batch_dict in enumerate(batch_generator):
        with torch.no_grad():
            logits_sent, logits_para, logits_Qtype = \
                            classifierGNN(batch_dict['feature_matrix'], batch_dict['adj'])

            max_value, max_index = logits_sent.max(dim=-1) # max_index is predict class.
            topN_sent_index_batch = (max_value * batch_dict['sent_mask'].squeeze()).topk(3, dim=-1)[1]
            topN_sent_index_batch = topN_sent_index_batch.squeeze().tolist()
            
        item = ques_items[index]
        info_list = [[item["node_list"][item["node_list"][s_id].parent_id].content_raw, 
                            item["node_list"][s_id].order_in_para,
                            item["node_list"][s_id].content_raw] \
                    for s_id in topN_sent_index_batch]

        sup_sent_id_list = [i[:-1] for i in info_list]
        sup_sent_list = [i[-1] for i in info_list]

        _values, indices = logits_Qtype.max(dim=-1)
        Qtype_list.append(indices.tolist()[0])
        
        sup_dict[item['id']] = sup_sent_id_list
        sup_raw_dict[item['id']] = sup_sent_list

        question = item["node_list"][0].content_raw
        QA_eval_list.append((question, sup_sent_list))

        pbar_GNN.update()

    # LM eval.
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    classifierQA = AutoQuestionAnswering.from_pretrained(model_path=args.model_path,
                                                        header_mode=args.header_mode,
                                                        cls_index=tokenizer.cls_token_id)
    classifierQA = classifierQA.to(args.device)
    checkpoint = torch.load(args.QA_model_path)
    classifierQA.load_state_dict(checkpoint['model'])
    _ = classifierQA.eval()
    classifierQA.to(args.device)

    datasetQA = HotpotQA_QA_Dataset.load_for_eval(QA_eval_list)
    datasetQA.set_parameters(tokenizer=tokenizer, topN_sents=args.topN_sents,
                            max_length=args.max_length, uncased=args.uncased,
                            permutations=False)
    batch_generatorQA = generate_QA_batches(datasetQA, 1, shuffle=False, drop_last=False, device=args.device)
    print(datasetQA)

    ans_dict = {}
    ans_dict_topN = defaultdict(list)

    pbar_QA = tqdm(total = datasetQA.get_num_batches(1), desc="QA EVAL")
    for index, batch_dict in enumerate(batch_generatorQA):
        with torch.no_grad():
            res = classifierQA(**batch_dict)
            start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits = res[:5]
            start_top_index = start_top_index.squeeze().tolist()
            end_top_index = end_top_index.squeeze().tolist()
            assert len(start_top_index) == len(end_top_index)
            
            input_ids = batch_dict['input_ids'].squeeze().tolist()
            item = ques_items[index]

            for index,(i,j) in enumerate(zip(start_top_index,end_top_index)):
                if index == 0:
                    if Qtype_list[index] == 0:
                        ans_dict[item['id']] = tokenizer.decode(input_ids[i:j+1])
                    else: # comparations
                        _values, indices = cls_logits.max(dim=-1)
                        ans = 'yes' if indices.tolist()[0] == 1 else 'no'
                        ans_dict[item['id']] = ans
                ans_dict_topN[item['id']].append(tokenizer.decode(input_ids[i:j+1]))
        pbar_QA.update()

    # combine.
    final_res = {}
    final_res['answer'] = ans_dict
    final_res['sp'] = sup_dict
    return final_res

def main(args):
    set_envs(args)
    with open(args.dev_json_path, 'r', encoding='utf-8') as f1:
        dev_json = json.load(f1)
    res = eval(args, dev_json)
    return res

def make_args():
    parser = argparse.ArgumentParser()

    # Data and model path.
    parser.add_argument(
        "--dev_json_path",
        default="data/HotpotQA/hotpot_dev_distractor_v1.json",
        type=str,help="remain",)
    parser.add_argument(
        "--GNN_model_path",
        default='save_model_GNN/GNN_HotpotQA_hidden64_heads8_pad300_chunk_first.pt',
        type=str,help="remain",)
    parser.add_argument(
        "--QA_model_path",
        default='save_model_QA_permutations/HotpotQA_QA_BiGRU_roberta-base-squad2.pt',
        type=str,help="remain",)
    parser.add_argument(
        "--model_path",
        default='data/models/roberta-base-squad2',
        type=str,help="remain",)

    # GNN parameters. MUST match saved pt file.
    parser.add_argument("--features",default=768,type=int,help="remain")
    parser.add_argument("--hidden",default=64,type=int,help="remain")
    parser.add_argument("--nclass",default=2,type=int,help="remain")
    parser.add_argument("--dropout",default=0.0,type=float,help="remain")
    parser.add_argument("--alpha",default=0.3,type=float,help="remain")
    parser.add_argument("--nheads",default=8,type=int,help="remain")
    parser.add_argument("--pad_max_num",default=300,type=int,help="remain")

    # Parameters for QA.
    parser.add_argument("--header_mode",default='MLP',type=str,help="remain")
    parser.add_argument("--cls_token_id",default=0,type=int,help="remain")
    parser.add_argument("--topN_sents",default=3,type=int,help="remain")
    parser.add_argument("--max_length",default=512,type=int,help="remain")
    parser.add_argument("--uncased", action="store_true", help="remain")

    # Runtime hyper parameter
    parser.add_argument("--cuda", action="store_true", help="remain")
    parser.add_argument("--device",default=None,type=str,help="remain")
    parser.add_argument("--do_eval", action="store_true", help="remain")

    # test setting
    parser.add_argument("--test_nums",default=-1,type=int,help="remain")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = make_args()
    res = main(args)
    time_now = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    with open(f"dev_distractor_pred_{time_now}.json", 'w', encoding='utf-8') as f1:
        json.dump(res, f1)
    if args.do_eval:
        import hotpot_evaluate_v1
        hotpot_evaluate_v1.eval(f"dev_distractor_pred_{time_now}.json", args.dev_json_path)
    
"""
test:
python evaluate.py --cuda \
    --hidden 256 --nheads 8 \
    --dev_json_path data/HotpotQA/hotpot_dev_distractor_v1.json \
    --GNN_model_path models_checkpoints/GNN/GNN_hidden256_heads8_pad300.pt  \
    --QA_model_path models_checkpoints/QA/HotpotQA_QA_MLP+unfreeze2_roberta-base.pt \
    --model_path data/models/roberta-base \
    --test_nums 3

formal:
python evaluate.py --cuda \
    --hidden 256 --nheads 8 \
    --dev_json_path data/HotpotQA/hotpot_dev_distractor_v1.json \
    --GNN_model_path models_checkpoints/GNN/GNN_hidden256_heads8_pad300.pt  \
    --QA_model_path models_checkpoints/QA/HotpotQA_QA_MLP+unfreeze2_roberta-base.pt \
    --model_path data/models/roberta-base \
    --do_eval 
"""
