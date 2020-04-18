# Fine-tuning Multi-hop Question Answering With Hierarchical Graph Network

a two stage model: model 1 is a graph neural network that reasons through message passing mechanism and selects support sentences as model output. the model 2 is a language model with a classifier header. 

## pre-requires

1. before run `pip install -r requirements.txt`, you need to download the latest package `[en_core_web_lg-2.2.5.tar.gz](https://github.com/explosion/spacy-models/releases//tag/en_core_web_lg-2.2.5)` and put it in `data/`.

2. download pre-trained language model weights from `[transformers library](https://huggingface.co/models)` and save the model in folder (with its name) in `data/models/`.

cache LM model code:

```python
from transformers import AutoModel, AutoTokenizer
model_name = 'bert-base-cased' # find model name in library.
foldername = 'bert-base-cased'
!mkdir -p data/models/$foldername
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(f"data/models/{foldername}/")
model = AutoModel.from_pretrained(model_name)
model.save_pretrained(f"data/models/{foldername}/")
```

## construct and initialize graph nodes

`gen_nodes_repr.py` contains all details. just specify one pre-trained LM (you need to cache it) and run:

```bash
nohup python gen_nodes_repr.py 
    --device cuda:0 --start 0 --end 90000 --split_num 200 
    --model_path data/models/bert-base-cased 
    --save_dir save_node_repr_bert-base-cased
    --spacy_model en_core_web_lg >> gen_nodes_repr_01.log &
```

**NOTICE** time consumption depends on the LM parameters number, but it usually takes a long time (1000 nodes representation generation under bert-base model would cost 8 hours). so set the start & end offet and use GPUs as more as possilbe. 

**NOTICE** output files (in `save_dir/`) may take a bulk of disk space (300GB+), please make sure enough space are available. 

## train GNN model

support single and multi GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 nohup python -m torch.distributed.launch train_GNN.py \
    --model_state_file GNN_hidden64_heads8_pad300.pt \
    --features 768 \
    --hidden 128 \
    --dropout 0 \
    --nheads 8 \
    --pad_max_num 300 \
    --topN_sents 3 \
    --num_epochs 2 \
    --chunk_size 100 \
    --batch_size 10 \
    --learning_rate 1e-3 \
    --save_dir save_model_GNN \
    --hotpotQA_item_folder save_node_repr_bert-base-cased \
    --log_dir parallel_runs_GNN/hidden64_heads8_pad300 \
    --cuda --fp16 >> train_GNN.log &
```


## train LM model

only support single GPU:

```bash
nohup python -m torch.distributed.launch train_QA.py --permutations \
    --model_state_file HotpotQA_QA_BiGRU_bert-base-cased.pt \
    --pretrained_model_path data/models/bert-base-cased \
    --log_dir parallel_runs_QA_permutations/BiGRU_bert-base-cased \
    --num_epochs 10 \
    --batch_size 8 \
    --cuda --fp16 >> train_QA.log &
```

## visualize comparation

use tensorboard:

```bash
tensorboard --logdir=parallel_runs_GNN/ --port 12345 --bind_all
tensorboard --logdir=parallel_runs_QA_permutations/ --port 12346 --bind_all
```

## evaluation

use the official evaluation script.

firstly, generate prediction file. the output file is something likes `dev_distractor_pred_2020-1-1_11:11:11.json`.

```bash
python evaluate.py \
    --dev_json_path data/HotpotQA/hotpot_dev_distractor_v1.json \
    --GNN_model_path save_model_GNN/GNN_HotpotQA_hidden64_heads8_pad300_chunk_first.pt \
    --QA_model_path save_model_QA_permutations/HotpotQA_QA_BiGRU_roberta-base-squad2.pt \
    --LMmodel_path data/models/roberta-base-squad2 
```

then run

```bash
python hotpot_evaluate_v1.py dev_distractor_pred_2020-1-1_11:11:11.json data/HotpotQA/hotpot_dev_distractor_v1.json
```