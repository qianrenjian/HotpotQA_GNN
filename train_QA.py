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

from tqdm import tqdm
from transformers import AutoTokenizer, AdamW
from apex import amp
from apex.parallel import DistributedDataParallel

from datasets import HotpotQA_QA_Dataset, find_ans_spans, generate_QA_batches
from QA_models import AutoQuestionAnswering
from utils import set_seed_everywhere, handle_dirs, make_train_state, update_train_state, get_linear_schedule_with_warmup
from traceback import print_exc

# args = Namespace(
#     # Data and path information
#     json_train_path='data/hotpot_train_v1.1.json',
#     use_mini=True,
#     json_train_mini_path='data/hotpot_train_mini.json',
#     model_state_file = "HotpotQA_QA_BiGRU_distilroberta-base-squad2.pt",
#     save_dir = 'save_cache_permutations',
#     pretrained_model_path = 'data/models/distilroberta-base-squad2',
#     freeze_layer_name='all',

#     # SummaryWriter
#     log_dir='runs_QA_permutations/BiGRU_distilroberta-base-squad2',
#     flush_secs=120,

#     # colab
#     colab=False,
#     colab_data_path='/content/drive/My Drive/DOWNLOAD/',
#     colab_project_path='/content/drive/My\ Drive/HotpotQA_XGM/',

#     # dataset parameters
#     uncased=False,
#     topN_sents=3,
#     max_length=512,
#     permutations=True,

#     # Training hyper parameter
#     num_epochs=1,
#     warmup_steps=0,
#     learning_rate=1e-3,
#     batch_size=12,
#     seed=666,
#     early_stopping_criteria=3,
#     weight_decay=0.01,
#     adam_epsilon=1e-8,

#     # Runtime hyper parameter
#     cuda=True,
#     device=None,
#     reload_from_files=False,
#     expand_filepaths_to_save_dir=True,
#     )

def set_envs(args):
    if not torch.cuda.is_available():
        args.cuda = False
        args.fp16 = False

    if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
        args.device_ids = eval(f"[{os.environ['CUDA_VISIBLE_DEVICES']}]")

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl',
                                        init_method='env://')
    torch.backends.cudnn.benchmark = True

    if not args.device:
        args.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if args.expand_filepaths_to_save_dir:
        args.model_state_file = os.path.join(args.save_dir,args.model_state_file)
    set_seed_everywhere(args.seed, args.cuda)
    handle_dirs(args.save_dir)
    
    if args.use_mini:
        args.json_train_path = args.json_train_mini_path
        args.log_dir = 'runs_mini'
    if args.colab:
        args.json_train_path = os.path.join(args.colab_data_path, args.json_train_path)
        args.json_train_mini_path = os.path.join(args.colab_data_path, args.json_train_mini_path)
        # args.model_state_file = os.path.join(args.colab_project_path, args.model_state_file)
        args.pretrained_model_path = os.path.join(args.colab_data_path, args.pretrained_model_path)
        # args.log_dir = os.path.join(args.colab_project_path, args.log_dir)
        from tqdm.notebook import tqdm

def compute_span_accuracy(start_logits, start_positions, end_logits, end_positions):

    _, start_indices = start_logits.max(-1)
    start_indices.eq(start_positions)

    _, end_indices = end_logits.max(-1)
    end_indices.eq(end_positions)

    correct = start_indices.eq(start_positions) * end_indices.eq(end_positions)

    numerator = correct.sum().item()
    denominator = start_positions.ne(-100).sum().item()

    # all questions are yes-no type.
    if denominator == 0: return None

    return float(numerator) / denominator

def compute_accuracy(logits, labels):

    _, logits_indices = logits.max(-1)    

    numerator = torch.eq(logits_indices, labels).sum().item()
    denominator = labels.ne(-100).sum().item()

    if denominator == 0: return None
    return float(numerator) / denominator

def main(args):
    set_envs(args)
    print("Using: {}".format(args.device))

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path, local_files_only=True)
    classifier = AutoQuestionAnswering.from_pretrained(model_path=args.pretrained_model_path,
                                                        cls_index=tokenizer.cls_token_id)
    classifier.freeze_to_layer_by_name(args.freeze_layer_name)
    classifier.train()

    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in classifier.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in classifier.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()),
    #                     lr=args.learning_rate)

    # Initialization
    opt_level = 'O1'
    if args.cuda:
        classifier = classifier.to(args.device)
        if args.fp16:
            classifier, optimizer = amp.initialize(classifier, optimizer, opt_level=opt_level)
        classifier = nn.parallel.DistributedDataParallel(classifier,
                                                        device_ids=args.device_ids, 
                                                        output_device=0, 
                                                        find_unused_parameters=True)

    if args.reload_from_files:
        checkpoint = torch.load(args.model_state_file)
        classifier.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        amp.load_state_dict(checkpoint['amp'])

    train_state = make_train_state(args)

    dataset = HotpotQA_QA_Dataset.build_dataset(args.json_train_path)
    dataset.set_parameters(tokenizer = tokenizer, topN_sents = args.topN_sents,
                            max_length=args.max_length, uncased=args.uncased,
                            permutations=args.permutations, random_seed=args.seed)
    print(dataset)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                            mode='min',
                            factor=0.7,
                            patience=dataset.get_num_batches(args.batch_size)/10)
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=args.warmup_steps,
    #     num_training_steps=dataset.get_num_batches(args.batch_size) * args.num_epochs
    # )

    try:
        writer = SummaryWriter(log_dir=args.log_dir,flush_secs=args.flush_secs)
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

        cursor_train = 0
        cursor_val = 0
        if args.reload_from_files and 'cursor_train' in checkpoint.keys():
            cursor_train = checkpoint['cursor_train'] + 1
            cursor_val = checkpoint['cursor_val'] + 1

        for epoch_index in range(args.num_epochs):

            train_state['epoch_index'] = epoch_index

            dataset.set_split('train')
            dataset.random_seed = args.seed + epoch_index
            batch_generator = generate_QA_batches(dataset,shuffle=False,
                                            batch_size=args.batch_size, 
                                            device=args.device)
            running_loss = 0.0
            running_ans_span_accuracy = 0.0
            running_yes_no_span_accuracy = 0.0

            classifier.train()

            # dont count running value if denominator == 0.
            batch_index_for_yesnospan = 0
            batch_index_for_span = 0

            for batch_index, batch_dict in enumerate(batch_generator):
                optimizer.zero_grad()
                yes_no_span = batch_dict.pop('yes_no_span')
                res = classifier(**batch_dict)
                start_logits, end_logits, cls_logits = res[0], res[1], res[2]
                
                start_loss = loss_fct(start_logits, batch_dict['start_positions'])
                end_loss = loss_fct(end_logits, batch_dict['end_positions'])
                start_end_loss = (start_loss + end_loss) / 2
                if start_end_loss > 1e5:
                    print(start_logits.gather(-1, batch_dict['start_positions'].view(-1, 1)))
                    print(batch_dict['special_tokens_mask'].gather(-1, batch_dict['start_positions'].view(-1, 1)))
                    print(batch_dict['start_positions'])
                    print('')
                    print(end_logits.gather(-1, batch_dict['end_positions'].view(-1, 1)))
                    print(batch_dict['special_tokens_mask'].gather(-1, batch_dict['end_positions'].view(-1, 1)))
                    print(batch_dict['end_positions'])
                    exit()

                yes_no_span_loss = loss_fct(cls_logits, yes_no_span) / 2
                if yes_no_span_loss > 1e5:
                    print(cls_logits)
                    print(yes_no_span)
                    exit()

                ans_span_accuracy = compute_span_accuracy(start_logits, batch_dict['start_positions'],
                                                            end_logits, batch_dict['end_positions'])
                yes_no_span_accuracy = compute_accuracy(cls_logits, yes_no_span)
                
                loss = start_end_loss + yes_no_span_loss
                running_loss += (loss.item() - running_loss) / (batch_index + 1)

                if ans_span_accuracy: 
                    running_ans_span_accuracy  += \
                                        (ans_span_accuracy - running_ans_span_accuracy) / (batch_index_for_span + 1)
                    batch_index_for_span += 1

                if yes_no_span_accuracy:
                    running_yes_no_span_accuracy  += \
                                        (yes_no_span_accuracy - running_yes_no_span_accuracy) / (batch_index_for_yesnospan + 1)
                    batch_index_for_yesnospan += 1
                
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                optimizer.step()
                scheduler.step(running_loss)  # Update learning rate schedule
                
                # update bar               
                train_bar.set_postfix(running_loss=running_loss,epoch=epoch_index)
                train_bar.update()

                writer.add_scalar('loss/train', loss.item(), cursor_train)
                if ans_span_accuracy:
                    writer.add_scalar('ans_span_accuracy/train', ans_span_accuracy, cursor_train)
                if yes_no_span_accuracy:
                    writer.add_scalar('yes_no_span_accuracy/train', yes_no_span_accuracy, cursor_train)

                writer.add_scalar('running_loss/train', running_loss, cursor_train)
                writer.add_scalar('running_ans_span_accuracy/train', running_ans_span_accuracy, cursor_train)
                writer.add_scalar('running_yes_no_span_accuracy/train', running_yes_no_span_accuracy, cursor_train)
                cursor_train += 1

            train_state['train_running_loss'].append(running_loss)

            # Iterate over val dataset
            # setup: batch generator, set loss and acc to 0; set eval mode on

            dataset.set_split('val')
            batch_generator = generate_QA_batches(dataset,
                                            batch_size=args.batch_size, 
                                            device=args.device)
            running_loss = 0.0
            running_ans_span_accuracy = 0.0
            running_yes_no_span_accuracy = 0.0
            
            classifier.eval()

            batch_index_for_yesnospan = 0
            batch_index_for_span = 0
            
            for batch_index, batch_dict in enumerate(batch_generator):
                with torch.no_grad():

                    yes_no_span = batch_dict.pop('yes_no_span')
                    res = classifier(**batch_dict)
                    start_logits, end_logits, cls_logits = res[0], res[1], res[2]

                    start_loss = loss_fct(start_logits, batch_dict['start_positions'])
                    end_loss = loss_fct(end_logits, batch_dict['end_positions'])
                    start_end_loss = (start_loss + end_loss) / 2
                    yes_no_span_loss = loss_fct(cls_logits, yes_no_span) / 2

                    ans_span_accuracy = compute_span_accuracy(start_logits, batch_dict['start_positions'],
                                                                end_logits, batch_dict['end_positions'])
                    yes_no_span_accuracy = compute_accuracy(cls_logits, yes_no_span)

                    loss = start_end_loss + yes_no_span_loss
                    running_loss += (loss.item() - running_loss) / (batch_index + 1)

                    if ans_span_accuracy: 
                        running_ans_span_accuracy  += \
                                            (ans_span_accuracy - running_ans_span_accuracy) / (batch_index_for_span + 1)
                        batch_index_for_span += 1
                    if yes_no_span_accuracy:
                        running_yes_no_span_accuracy  += \
                                            (yes_no_span_accuracy - running_yes_no_span_accuracy) / (batch_index_for_yesnospan + 1)
                        batch_index_for_yesnospan += 1

                val_bar.set_postfix(running_loss=running_loss,epoch=epoch_index)
                val_bar.update()

                writer.add_scalar('loss/val', loss.item(), cursor_val)
                if ans_span_accuracy:
                    writer.add_scalar('ans_span_accuracy/val', ans_span_accuracy, cursor_val)
                if yes_no_span_accuracy:
                    writer.add_scalar('yes_no_span_accuracy/val', yes_no_span_accuracy, cursor_val)

                writer.add_scalar('running_loss/val', running_loss, cursor_val)
                writer.add_scalar('running_ans_span_accuracy/val', running_ans_span_accuracy, cursor_val)
                writer.add_scalar('running_yes_no_span_accuracy/val', running_yes_no_span_accuracy, cursor_val)
                cursor_val += 1

            train_state['val_running_loss'].append(running_loss)

            if not args.use_mini:
                train_state = update_train_state(args=args,
                                                model=classifier, 
                                                optimizer = optimizer,
                                                train_state=train_state)

            train_bar.n = 0
            val_bar.n = 0
            epoch_bar.update()

            if train_state['stop_early']:
                print('STOP EARLY!')
                break

    except KeyboardInterrupt:
        print("Exiting loop")
        os.remove(args.log_dir)
    except :
        print_exc()
        print(f"err in epoch_index {epoch_index}, batch_index {batch_index}.")

def make_args():
    parser = argparse.ArgumentParser()

    # Data and path information
    parser.add_argument(
        "--json_train_path",
        default='data/hotpot_train_v1.1.json',
        type=str,
        help="remain",
            )
    parser.add_argument("--use_mini", action="store_true", help="remain")
    parser.add_argument(
        "--json_train_mini_path",
        default='data/hotpot_train_mini.json',
        type=str,
        help="remain",
            )
    parser.add_argument(
        "--model_state_file",
        default="HotpotQA_QA_BiGRU_distilroberta-base-squad2.pt",
        type=str,
        help="remain",
            )
    parser.add_argument(
        "--save_dir",
        default='save_cache_permutations',
        type=str,
        help="remain",
            )
    parser.add_argument(
        "--pretrained_model_path",
        default='data/models/distilroberta-base-squad2',
        type=str,
        help="remain",
            )
    parser.add_argument("--freeze_layer_name",default='all',type=str,help="remain")

    # SummaryWriter
    parser.add_argument(
        "--log_dir",
        default='runs_QA_permutations/BiGRU_distilroberta-base-squad2',
        type=str,
        help="remain",
            )
    parser.add_argument("--flush_secs",default=120,type=int,help="remain",)

    # colab
    parser.add_argument("--model_type",default=None,type=str,help="remain",)
    parser.add_argument("--colab", action="store_true", help="remain")
    parser.add_argument(
        "--colab_data_path",
        default='/content/drive/My Drive/DOWNLOAD/',
        type=str,
        help="remain",
            )
    parser.add_argument(
        "--colab_project_path",
        default='/content/drive/My\ Drive/HotpotQA_XGM/',
        type=str,
        help="remain",
            )

    # Dataset parameters
    parser.add_argument("--uncased", action="store_true", help="remain")
    parser.add_argument("--topN_sents",default=3,type=int,help="remain",)
    parser.add_argument("--max_length",default=512,type=int,help="remain",)
    parser.add_argument("--permutations", action="store_true", help="remain")

    # Training hyper parameter
    parser.add_argument("--num_epochs",default=1,type=int,help="remain",)
    parser.add_argument("--batch_size",default=12,type=int,help="remain",)
    parser.add_argument("--warmup_steps",default=0,type=int,help="remain",)
    parser.add_argument("--learning_rate",default=1e-3,type=float,help="remain",)
    parser.add_argument("--seed",default=666,type=int,help="remain",)
    parser.add_argument("--early_stopping_criteria",default=3,type=int,help="remain",)
    parser.add_argument("--weight_decay",default=0.01,type=float,help="remain",)
    parser.add_argument("--adam_epsilon",default=1e-8,type=float,help="remain",)

    # Runtime hyper parameter
    parser.add_argument("--cuda", action="store_true", help="remain")
    parser.add_argument("--device",default=None,help="remain",)
    parser.add_argument("--reload_from_files", action="store_true", help="remain")
    parser.add_argument("--expand_filepaths_to_save_dir",default=True,help="remain",)
    parser.add_argument("--fp16", action="store_true", help="remain")

    # Data parallel setting
    parser.add_argument('--local_rank', metavar='int', type=int, default=0, help='rank')
    parser.add_argument("--dbp_port",default=23456,type=int,help="remain",)
    parser.add_argument("--device_ids",default=[0],type=list,help="remain",)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = make_args()
    main(args)

"""
test
python -m torch.distributed.launch train_QA.py --use_mini --uncased --permutations \
    --cuda \
    --model_state_file HotpotQA_QA_BiGRU_distilroberta-base-squad2.pt \
    --pretrained_model_path data/models/distilroberta-base-squad2 \
    --log_dir parallel_runs_QA_permutations/BiGRU_distilroberta-base-squad2 \
    --num_epochs 1 \
    --batch_size 12 
"""