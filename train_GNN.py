import os
import sys
from argparse import Namespace
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import scipy.sparse as sp
import ujson as json
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer
from apex import amp

from datasets import HotpotQA_GNN_Dataset, gen_GNN_batches
from GNN import GAT_HotpotQA
from utils import set_seed_everywhere, handle_dirs, make_train_state, update_train_state
from traceback import print_exc

def compute_accuracy(logits=None, labels=None, predict=None):
    if predict == None: _, predict = logits.max(dim=1)
    n_correct = torch.eq(predict, labels).sum().item()
    return float(n_correct) / len(predict)

def compute_recall(logits, labels, mask):
    '''only count the positive recall'''
    _, logits_indices = logits.max(dim=1)
    all_positive_predicts = (logits_indices * labels * mask).sum().item() # only positive
    all_positive_labels = (labels * mask).sum().item()
    return float(all_positive_predicts) / all_positive_labels

# args = Namespace(
#     # Data and path information
#     model_state_file = "GNN_hidden64_heads8_pad300_chunk_first.pt",
#     save_dir = 'save_cache_GNN',
#     hotpotQA_item_folder = 'save_preprocess_new',
#     log_dir='runs_GNN/hidden64_heads8_pad300_chunk_first',

#     # Dataset parameter
#     pad_max_num = 300,
#     pad_value = 0,

#     # Training hyper parameter
#     chunk_size = 15000,
#     num_epochs=3,
#     learning_rate=1e-3,
#     batch_size=24,
#     topN_sents=4,
#     seed=1337,
#     early_stopping_criteria=5,
#     flush_secs=60,

#     # GNN parameters
#     features = 768,
#     hidden = 64,
#     nclass = 2,
#     dropout = 0,
#     alpha = 0.3,
#     nheads = 8,
    
#     # Runtime hyper parameter
#     cuda=True,
#     device=None,
#     reload_from_files=False,
#     expand_filepaths_to_save_dir=True,
#     )

def main(args):
    if not torch.cuda.is_available():
        args.cuda = False
    if not args.device:
        args.device = torch.device("cuda" if args.cuda else "cpu")
    if args.expand_filepaths_to_save_dir:
        args.model_state_file = os.path.join(args.save_dir,args.model_state_file)
    
    set_seed_everywhere(args.seed, args.cuda)
    handle_dirs(args.save_dir)
    print("Using: {}".format(args.device))

    total_items = os.popen(f'ls -l {args.hotpotQA_item_folder} |grep "^-"| wc -l').readlines()[0].strip()
    total_items = int(total_items)
    print(f"total items: {total_items}")

    # 实例化
    classifier = GAT_HotpotQA(features=args.features, hidden=args.hidden, nclass=args.nclass, 
                                dropout=args.dropout, alpha=args.alpha, nheads=args.nheads, 
                                nodes_num=args.pad_max_num)

    class_weights_sent, class_weights_para, class_weights_Qtype = \
                HotpotQA_GNN_Dataset.get_weights(device=args.device)
    loss_func_sent = nn.CrossEntropyLoss(class_weights_sent,ignore_index=-100)
    loss_func_para = nn.CrossEntropyLoss(class_weights_para,ignore_index=-100)
    loss_func_Qtype = nn.CrossEntropyLoss(class_weights_Qtype)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()),
                        lr=args.learning_rate)

    # Initialization
    opt_level = 'O1'
    if args.cuda:
        classifier = classifier.cuda()
        if args.fp16: classifier, optimizer = amp.initialize(classifier, optimizer, opt_level=opt_level)
        torch.distributed.init_process_group(backend="nccl")
        classifier = nn.parallel.DistributedDataParallel(classifier)

    if args.reload_from_files:
        checkpoint = torch.load(args.model_state_file)
        classifier.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if args.fp16: amp.load_state_dict(checkpoint['amp'])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,\
                                                    mode='min', factor=0.7, patience=30)
    train_state = make_train_state(args)

    try:
        writer = SummaryWriter(log_dir=args.log_dir, flush_secs=args.flush_secs)
        cursor_train = 0
        cursor_val = 0
        # for epoch_index in range(args.num_epochs):

        for chunk_i in range(0, total_items, args.chunk_size):
            dataset = HotpotQA_GNN_Dataset.build_dataset(hotpotQA_item_folder = args.hotpotQA_item_folder,
                                                        i_from = chunk_i, 
                                                        i_to = chunk_i+args.chunk_size,
                                                        seed=args.seed+chunk_i)
            dataset.set_parameters(args.pad_max_num, args.pad_value)

            epoch_bar = tqdm(desc='training routine',
                            total=args.num_epochs,
                            position=0)

            dataset.set_split('train')
            train_bar = tqdm(desc='split=train',
                            total=dataset.get_num_batches(args.batch_size), 
                            position=1)

            dataset.set_split('val')
            val_bar = tqdm(desc='split=val',
                            total=dataset.get_num_batches(args.batch_size), 
                            position=1)

            for epoch_index in range(args.num_epochs):

                train_state['epoch_index'] = epoch_index

                dataset.set_split('train')
                batch_generator = gen_GNN_batches(dataset,
                                                batch_size=args.batch_size, 
                                                device=args.device)
                running_loss = 0.0
                running_acc_Qtype = 0.0
                running_acc_topN = 0.0

                classifier.train()

                for batch_index, batch_dict in enumerate(batch_generator):

                    optimizer.zero_grad()

                    logits_sent, logits_para, logits_Qtype = \
                                    classifier(batch_dict['feature_matrix'], batch_dict['adj'])
                    print(f"logits_sent: {logits_sent}")

                    # topN sents
                    max_value, max_index = logits_sent.max(dim=-1) # max_index is predict class.
                    topN_sent_index_batch = (max_value * batch_dict['sent_mask'].squeeze()).topk(args.topN_sents, dim=-1)[1]
                    topN_sent_predict = torch.gather(max_index, -1, topN_sent_index_batch)
                    topN_sent_label = torch.gather((batch_dict['labels'] * batch_dict['sent_mask']).squeeze(),
                                                    -1, 
                                                    topN_sent_index_batch)

                    logits_sent = (batch_dict['sent_mask'] * logits_sent).view(-1,2)

                    labels_sent = (batch_dict['sent_mask']*batch_dict['labels'] + \
                                batch_dict['sent_mask'].eq(0)*-100).view(-1)

                    logits_para = (batch_dict['para_mask'] * logits_para).view(-1,2)

                    labels_para = (batch_dict['para_mask']*batch_dict['labels'] + \
                                batch_dict['para_mask'].eq(0)*-100).view(-1)

                    loss_sent = loss_func_sent(logits_sent, labels_sent) # [B,2] [B]
                    loss_para = loss_func_para(logits_para, labels_para) # [B,2] [B]
                    loss_Qtype = loss_func_Qtype(logits_Qtype.view(-1,2),
                                                batch_dict['answer_type'].view(-1)) # [B,2] [B]

                    loss = loss_sent + loss_para + loss_Qtype
                    print(f"loss:{loss}")
                    running_loss += (loss.item() - running_loss) / (batch_index + 1)

                    if args.fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()

                    optimizer.step()
                    scheduler.step(running_loss)

                    # compute the recall
                    recall_t_sent = compute_recall(logits_sent.view(-1,2), 
                                                batch_dict['labels'].view(-1), 
                                                batch_dict['sent_mask'].view(-1))

                    recall_t_para = compute_recall(logits_para.view(-1,2), 
                                                batch_dict['labels'].view(-1), 
                                                batch_dict['para_mask'].view(-1))
                    # compute the acc
                    acc_t_Qtype = compute_accuracy(logits_Qtype.view(-1,2), 
                                                batch_dict['answer_type'].view(-1))
                    running_acc_Qtype += (acc_t_Qtype - running_acc_Qtype) / (batch_index + 1)

                    acc_t_topN = compute_accuracy(predict=topN_sent_predict.view(-1), 
                                                labels=topN_sent_label.view(-1))
                    running_acc_topN += (acc_t_topN - running_acc_topN) / (batch_index + 1)


                    # update bar
                    train_bar.set_postfix(loss=running_loss,
                                        epoch=epoch_index)
                    train_bar.update()
                    
                    writer.add_scalar('loss/train', loss.item(), cursor_train)
                    writer.add_scalar('recall_t_sent/train', recall_t_sent, cursor_train)
                    writer.add_scalar('recall_t_para/train', recall_t_para, cursor_train)

                    writer.add_scalar('running_acc_Qtype/train', running_acc_Qtype, cursor_train)
                    writer.add_scalar('running_acc_topN/train', running_acc_topN, cursor_train)
                    writer.add_scalar('running_loss/train', running_loss, cursor_train)
                    cursor_train += 1

                train_state['train_running_loss'].append(running_loss)

                # Iterate over val dataset
                # setup: batch generator, set loss and acc to 0; set eval mode on

                dataset.set_split('val')
                batch_generator = gen_GNN_batches(dataset,
                                                batch_size=args.batch_size, 
                                                device=args.device)
                running_loss = 0.0
                running_acc_Qtype = 0.0
                running_acc_topN = 0.0
                classifier.eval()

                for batch_index, batch_dict in enumerate(batch_generator):
                    # compute the output
                    with torch.no_grad():

                        logits_sent, logits_para, logits_Qtype = \
                                        classifier(batch_dict['feature_matrix'], batch_dict['adj'])

                        # topN sents
                        max_value, max_index = logits_sent.max(dim=-1) # max_index is predict class.
                        topN_sent_index_batch = (max_value * batch_dict['sent_mask'].squeeze()).topk(args.topN_sents, dim=-1)[1]
                        topN_sent_predict = torch.gather(max_index, -1, topN_sent_index_batch)
                        topN_sent_label = torch.gather((batch_dict['labels'] * batch_dict['sent_mask']).squeeze(),
                                                        -1, 
                                                        topN_sent_index_batch)

                        logits_sent = (batch_dict['sent_mask'] * logits_sent).view(-1,2)

                        labels_sent = (batch_dict['sent_mask']*batch_dict['labels'] + \
                                    batch_dict['sent_mask'].eq(0)*-100).view(-1)

                        logits_para = (batch_dict['para_mask'] * logits_para).view(-1,2)

                        labels_para = (batch_dict['para_mask']*batch_dict['labels'] + \
                                    batch_dict['para_mask'].eq(0)*-100).view(-1)

                        loss_sent = loss_func_sent(logits_sent, labels_sent) # [B,2] [B]
                        loss_para = loss_func_para(logits_para, labels_para) # [B,2] [B]
                        loss_Qtype = loss_func_Qtype(logits_Qtype.view(-1,2),
                                                    batch_dict['answer_type'].view(-1)) # [B,2] [B]

                        loss = loss_sent + loss_para + loss_Qtype
                        running_loss += (loss.item() - running_loss) / (batch_index + 1)

                    # compute the recall
                    recall_t_sent = compute_recall(logits_sent.view(-1,2), 
                                                batch_dict['labels'].view(-1), 
                                                batch_dict['sent_mask'].view(-1))

                    recall_t_para = compute_recall(logits_para.view(-1,2), 
                                                batch_dict['labels'].view(-1), 
                                                batch_dict['para_mask'].view(-1))
                    # compute the acc
                    acc_t_Qtype = compute_accuracy(logits_Qtype.view(-1,2), 
                                                batch_dict['answer_type'].view(-1))
                    running_acc_Qtype += (acc_t_Qtype - running_acc_Qtype) / (batch_index + 1)

                    acc_t_topN = compute_accuracy(predict=topN_sent_predict.view(-1), 
                                                labels=topN_sent_label.view(-1))
                    running_acc_topN += (acc_t_topN - running_acc_topN) / (batch_index + 1)


                    # update bar
                    val_bar.set_postfix(loss=running_loss,
                                        epoch=epoch_index)
                    val_bar.update()
                    
                    writer.add_scalar('loss/val', loss.item(), cursor_val)
                    writer.add_scalar('recall_t_sent/val', recall_t_sent, cursor_val)
                    writer.add_scalar('recall_t_para/val', recall_t_para, cursor_val)

                    writer.add_scalar('running_acc_Qtype/val', running_acc_Qtype, cursor_val)
                    writer.add_scalar('running_acc_topN/val', running_acc_topN, cursor_val)
                    writer.add_scalar('running_loss/val', running_loss, cursor_val)
                    cursor_val += 1


                train_state['val_running_loss'].append(running_loss)
                writer.add_scalar('running_loss/val', running_loss, cursor_val)

                train_state = update_train_state(args=args, model=classifier, 
                                                optimizer = optimizer,
                                                train_state=train_state)

                train_bar.n = 0
                val_bar.n = 0
                epoch_bar.update()

                if train_state['stop_early']:
                    break

    except KeyboardInterrupt:
        print("Exiting loop")
    except :
        print(f"err in chunk {chunk_i}, epoch_index {epoch_index}, batch_index {batch_index}.")
        print_exc()

def make_args():
    parser = argparse.ArgumentParser()

    # Data and path information
    parser.add_argument(
        "--model_state_file",
        default="GNN_hidden64_heads8_pad300_chunk_first.pt",
        type=str,
        help="remain",
            )
    parser.add_argument(
        "--save_dir",
        default='save_cache_GNN',
        type=str,
        help="remain",
            )
    parser.add_argument(
        "--hotpotQA_item_folder",
        default='save_preprocess_new',
        type=str,
        help="remain",
            )
    parser.add_argument(
        "--log_dir",
        default='runs_GNN/hidden64_heads8_pad300_chunk_first',
        type=str,
        help="remain",
            )

    # Dataset parameter
    parser.add_argument("--pad_max_num",default=300,type=int,help="remain")
    parser.add_argument("--pad_value",default=0,type=int,help="remain")

    # Training hyper parameter
    parser.add_argument("--chunk_size",default=15000,type=int,help="remain")
    parser.add_argument("--num_epochs",default=3,type=int,help="remain")
    parser.add_argument("--learning_rate",default=1e-3,type=float,help="remain")
    parser.add_argument("--batch_size",default=24,type=int,help="remain")
    parser.add_argument("--topN_sents",default=3,type=int,help="remain")
    parser.add_argument("--seed",default=0,type=int,help="remain")
    parser.add_argument("--early_stopping_criteria",default=300,type=int,help="remain")
    parser.add_argument("--flush_secs",default=0,type=int,help="remain")

    # GNN parameters
    parser.add_argument("--features",default=768,type=int,help="remain")
    parser.add_argument("--hidden",default=64,type=int,help="remain")
    parser.add_argument("--nclass",default=2,type=int,help="remain")
    parser.add_argument("--dropout",default=0,type=float,help="remain")
    parser.add_argument("--alpha",default=0.3,type=float,help="remain")
    parser.add_argument("--nheads",default=8,type=int,help="remain",)

    # Runtime hyper parameter
    parser.add_argument("--cuda", action="store_true", help="remain")
    parser.add_argument("--device",default=None,help="remain",)
    parser.add_argument("--reload_from_files", action="store_true", help="remain")
    parser.add_argument("--expand_filepaths_to_save_dir", action="store_true", help="remain")
    parser.add_argument("--fp16", action="store_true", help="remain")

    # Data parallel setting
    parser.add_argument("--gpu0_bsz",default=6,type=int,help="remain",)
    parser.add_argument("--acc_grad",default=1,type=int,help="remain",)
    parser.add_argument('--local_rank', metavar='int', type=int, dest='rank', default=0, help='rank')
    parser.add_argument("--dbp_port",default=23456,type=int,help="remain",)
    parser.add_argument("--visible_devices",default='0',type=str,help="remain",)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = make_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.visible_devices
    main(args)

"""
test
python -m torch.distributed.launch train_GNN.py --cuda --expand_filepaths_to_save_dir \
    --reload_from_files \
    --model_state_file GNN_hidden64_heads8_pad300_chunk_first.pt \
    --save_dir save_cache_GNN \
    --hotpotQA_item_folder save_preprocess_new \
    --log_dir parallel_runs_GNN/hidden64_heads8_pad300_chunk_first \
    --visible_devices 0,1
    --chunk_size 100
"""