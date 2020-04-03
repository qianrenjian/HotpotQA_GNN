

import argparse

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
        default="HotpotQA_QA_BiGRU_bert-base-cased.pt",
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
        default='data/models/bert-base-cased',
        type=str,
        help="remain",
            )
    parser.add_argument(
        "--freeze_layer_name",
        default='all',
        type=str,
        help="remain",
            )

    # SummaryWriter
    parser.add_argument(
        "--log_dir",
        default='runs_QA_permutations/BiGRU_bert-base-cased',
        type=str,
        help="remain",
            )
    parser.add_argument(
        "--flush_secs",
        default=120,
        type=int,
        help="remain",
            )

    # colab
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        help="remain",
            )
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

    # dataset parameters
    parser.add_argument("--uncased", action="store_true", help="remain")
    parser.add_argument(
        "--topN_sents",
        default=3,
        type=int,
        help="remain",
            )
    parser.add_argument(
        "--max_length",
        default=512,
        type=int,
        help="remain",
            )
    parser.add_argument("--permutations", action="store_true", help="remain")

    # Training hyper parameter
    parser.add_argument(
        "--num_epochs",
        default=1,
        type=int,
        help="remain",
            )
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="remain",
            )
    parser.add_argument(
        "--learning_rate",
        default=1e-3,
        type=float,
        help="remain",
            )
    parser.add_argument(
        "--batch_size",
        default=12,
        type=int,
        help="remain",
            )
    parser.add_argument(
        "--seed",
        default=666,
        type=int,
        help="remain",
            )
    parser.add_argument(
        "--early_stopping_criteria",
        default=3,
        type=int,
        help="remain",
            )
    parser.add_argument(
        "--weight_decay",
        default=0.01,
        type=float,
        help="remain",
            )
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="remain",
            )

    # Runtime hyper parameter
    parser.add_argument("--cuda", action="store_true", help="remain")
    parser.add_argument(
        "--device",
        default=None,
        help="remain",
            )
    parser.add_argument("--reload_from_files", action="store_true", help="remain")
    parser.add_argument("--expand_filepaths_to_save_dir", action="store_true", help="remain")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = make_args()
    print(args)

sleep 1.5h
# 1
python train_QA.py --permutations --cuda --expand_filepaths_to_save_dir --reload_from_files \
--model_state_file HotpotQA_QA_BiGRU_bert-base-cased.pt \
--pretrained_model_path data/models/bert-base-cased \
--log_dir runs_QA_permutations/BiGRU_unfreeze_1_bert-base-cased \
--freeze_layer_name encoder.layer.11 \
--num_epochs 2 \
--batch_size 6 \
--learning_rate 3e-5

# 2
python train_QA.py --permutations --cuda --expand_filepaths_to_save_dir --reload_from_files \
--model_state_file HotpotQA_QA_BiGRU_bert-base-cased.pt \
--pretrained_model_path data/models/bert-base-cased \
--log_dir runs_QA_permutations/BiGRU_unfreeze_2_bert-base-cased \
--freeze_layer_name encoder.layer.10 \
--num_epochs 1 \
--batch_size 4 \
--learning_rate 3e-5

# 3
python train_QA.py --permutations --cuda --expand_filepaths_to_save_dir --reload_from_files \
--model_state_file HotpotQA_QA_BiGRU_bert-base-cased.pt \
--pretrained_model_path data/models/bert-base-cased \
--log_dir runs_QA_permutations/BiGRU_unfreeze_3_bert-base-cased \
--freeze_layer_name encoder.layer.9 \
--num_epochs 1 \
--batch_size 2 \
--learning_rate 3e-5

# backup
python train_QA.py --uncased --use_mini --permutations --cuda --expand_filepaths_to_save_dir \
--model_state_file HotpotQA_QA_BiGRU_bert-base-cased.pt \
--pretrained_model_path data/models/bert-base-cased \
--log_dir runs_QA_permutations/BiGRU_unfreeze_1_bert-base-cased \
--freeze_layer_name encoder.layer.11 \
--num_epochs 1 \
--batch_size 7 \
--learning_rate 1e-4


### train GNN
